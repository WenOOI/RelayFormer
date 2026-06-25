"""
Mixed image / video dataset for RelayFormer manipulation localization.

A single dataset class drives all three training modes:
  - pure image  : pass only `image_datasets`
  - pure video  : pass only `video_datasets`
  - mixed       : pass both

Images are treated as single-frame clips (clip_len=1); videos as clips of `clip_len`
frames. `collate_fn` flattens everything into a frame batch plus a per-sample `clip_len`
vector, which is exactly what `RelayFormer.forward` expects.

Config format (list of entries):
  ["ManiDataset", "/path/to/dir"]                 # image dir with Tp/ and Gt/
  ["JsonDataset", "/path/to/list.json"]           # image list: [[tp, gt], ...]
  ["VideoDataset", "/path/to/dir", ["m1", "m2"]]  # video dir + manipulation methods
"""
import os
import json
from glob import glob
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class EdgeMaskGenerator(torch.nn.Module):
    """Generate the 'edge bar' for a binary mask using morphological dilation and difference."""
    def __init__(self, kernel_size: int = 3) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        k = torch.zeros((1, 1, kernel_size, kernel_size), dtype=torch.float32)
        center = kernel_size // 2
        k[0, 0, center, :] = 1
        k[0, 0, :, center] = 1
        self.register_buffer('kernel', k)

    def _dilate(self, x: torch.Tensor) -> torch.Tensor:
        return (F.conv2d(x, self.kernel, padding=self.kernel_size // 2) > 0).float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = (x > 0).float()
        dilated = self._dilate(x)
        eroded = 1 - self._dilate(1 - x)
        edge = ((dilated - eroded).abs() > 0).float()
        return edge.squeeze(1)  # (B, H, W)


def pil_loader(path):
    """PIL image loader.

    Open via a file handle that is closed before `convert()` decodes (matches
    IMDLBenCo's loader).  Avoids PIL's lazy-fd behaviour which causes repeated
    syscalls and worker contention under multi-process DataLoader.
    """
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_albu_transforms(type_: str, output_size: Tuple[int, int]):
    """Padding or resizing transform applied after augmentation."""
    if type_ == "pad":
        return A.Compose([
            A.PadIfNeeded(
                min_height=output_size[0],
                min_width=output_size[1],
                border_mode=0,
                value=0,
                position='top_left',
                mask_value=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])
    elif type_ == "resize":
        return A.Compose([
            A.Resize(output_size[0], output_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.Crop(0, 0, output_size[0], output_size[1]),
            ToTensorV2(transpose_mask=True)
        ])
    else:
        raise ValueError(f"Unsupported transform type: {type_}")


class MixedImageVideoDataset(Dataset):
    """Mixed image + video dataset configured via lists of dataset entries."""

    def __init__(self,
                 image_datasets: Optional[List[List[str]]] = None,   # [["ManiDataset"/"JsonDataset", path], ...]
                 video_datasets: Optional[List[List[str]]] = None,   # [["VideoDataset", path, methods], ...]
                 split: str = 'train',
                 clip_len: int = 4,
                 is_padding: bool = False,
                 is_resizing: bool = False,
                 output_size: Tuple[int, int] = (1024, 1024),
                 edge_width: Optional[int] = None,
                 img_loader=pil_loader,
                 post_funcs=None,
                 image_transforms=None,   # albumentations transform for images
                 video_transforms=None,   # albumentations transform for video frames
                 ) -> None:
        super().__init__()

        if is_padding and is_resizing:
            raise AttributeError("is_padding and is_resizing can not be True at the same time")
        if not is_padding and not is_resizing:
            raise AttributeError("is_padding and is_resizing can not be False at the same time")
        if image_datasets is None and video_datasets is None:
            raise ValueError("At least one of image_datasets or video_datasets must be provided")

        self.split = split
        self.clip_len = clip_len
        self.is_padding = is_padding
        self.is_resizing = is_resizing
        self.output_size = output_size
        self.img_loader = img_loader
        self.post_funcs = post_funcs

        # Samples grouped per source dataset.
        self.image_datasets_samples: List[List[Dict]] = []
        self.video_datasets_samples: List[List[Dict]] = []
        self.image_dataset_names: List[str] = []
        self.video_dataset_names: List[str] = []

        self._init_datasets(image_datasets, video_datasets)
        self._setup_transforms(image_transforms, video_transforms)
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)
        self._setup_sampling()

        total_image_samples = sum(len(s) for s in self.image_datasets_samples)
        total_video_samples = sum(len(s) for s in self.video_datasets_samples)
        print("Dataset initialized:")
        print(f"  Image datasets: {len(self.image_datasets_samples)}, total samples: {total_image_samples}")
        print(f"  Video datasets: {len(self.video_datasets_samples)}, total samples: {total_video_samples}")
        print(f"  Epoch length: {self._len}")

    # ------------------------------------------------------------------ init
    def _init_datasets(self, image_datasets, video_datasets):
        if image_datasets is not None:
            for dataset_config in image_datasets:
                dataset_type, dataset_path = dataset_config[0], dataset_config[1]
                print(f"Loading image dataset: {dataset_type} from {dataset_path}")
                dataset_samples: List[Dict] = []
                if dataset_type == "ManiDataset":
                    self._init_mani_dataset(dataset_path, dataset_samples)
                elif dataset_type == "JsonDataset":
                    self._init_json_dataset(dataset_path, dataset_samples)
                else:
                    raise ValueError(f"Unknown image dataset type: {dataset_type}")
                self.image_datasets_samples.append(dataset_samples)
                self.image_dataset_names.append(f"{dataset_type}:{dataset_path}")

        if video_datasets is not None:
            for dataset_config in video_datasets:
                dataset_type, dataset_path, video_methods = dataset_config
                print(f"Loading video dataset: {dataset_type} from {dataset_path}")
                dataset_samples = []
                self._init_video_dataset(dataset_path, video_methods, dataset_samples)
                self.video_datasets_samples.append(dataset_samples)
                self.video_dataset_names.append(f"{dataset_type}:{dataset_path}")

    def _init_mani_dataset(self, dataset_path: str, dataset_samples: List):
        """ManiDataset image folder: <path>/Tp (tampered) and <path>/Gt (ground truth)."""
        tp_dir = os.path.join(dataset_path, 'Tp')
        gt_dir = os.path.join(dataset_path, 'Gt')
        if not (os.path.exists(tp_dir) and os.path.exists(gt_dir)):
            print(f"Warning: ManiDataset directories not found: {tp_dir} or {gt_dir}")
            return

        exts = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
        tp_files = sorted(f for f in os.listdir(tp_dir) if f.lower().endswith(exts))
        gt_files = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(exts))
        for i in range(min(len(tp_files), len(gt_files))):
            tp_path = os.path.join(tp_dir, tp_files[i])
            gt_path = os.path.join(gt_dir, gt_files[i])
            if os.path.exists(tp_path) and os.path.exists(gt_path):
                dataset_samples.append({
                    'type': 'image', 'dataset_type': 'ManiDataset',
                    'dataset_path': dataset_path, 'tp_path': tp_path, 'gt_path': gt_path,
                })
        print(f"Loaded {len(dataset_samples)} samples from ManiDataset: {dataset_path}")

    def _init_json_dataset(self, json_path: str, dataset_samples: List):
        """JsonDataset: a JSON list of [tampered_path, gt_path] pairs ("Negative" gt allowed)."""
        if not os.path.exists(json_path):
            print(f"Warning: JSON file not found: {json_path}")
            return
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        count = 0
        for item in json_data:
            tp_path, gt_path = item[0], item[1]
            if os.path.isfile(tp_path) and (gt_path == "Negative" or os.path.isfile(gt_path)):
                dataset_samples.append({
                    'type': 'image', 'dataset_type': 'JsonDataset',
                    'dataset_path': json_path, 'tp_path': tp_path, 'gt_path': gt_path,
                })
                count += 1
            else:
                raise TypeError(f"Get Error when loading from {item}")
        print(f"Loaded {count} samples from JsonDataset: {json_path}")

    def _init_video_dataset(self, video_path: str, video_methods: List[str], dataset_samples: List):
        """
        Build video clip samples. Two on-disk layouts are supported:
          - MOSE / Splice style: <path>/<method>/<frame_dir>/<video>/*.png with parallel mask dir
          - generic style:       <path>/fake/<method>/<video>/*.jpg with <path>/mask/<video>/*
        """
        if 'MOSE' in video_path or 'Splice' in video_path:
            datatype = 'MOSE' if 'MOSE' in video_path else 'Splice'
            fake = 'frame' if datatype == 'MOSE' else 'fake'
            mask_name = 'groundtruth' if datatype == 'MOSE' else 'mask'

            for method in video_methods:
                fake_dir = os.path.join(video_path, method, fake)
                if not os.path.isdir(fake_dir):
                    continue
                for video in sorted(os.listdir(fake_dir)):
                    frame_dir = os.path.join(fake_dir, video)
                    if not os.path.isdir(frame_dir):
                        continue
                    frames = sorted(glob(os.path.join(frame_dir, '*.png')))
                    if len(frames) < self.clip_len:
                        continue

                    if self.split == 'test':
                        starts = [0, self.clip_len - 1, max(0, len(frames) - self.clip_len)]
                    else:
                        starts = list(range(0, max(0, len(frames) - self.clip_len), self.clip_len))

                    for i in starts:
                        clip = frames[i:i + self.clip_len]
                        if len(clip) != self.clip_len:
                            continue
                        if datatype == 'MOSE':
                            masks = [os.path.join(video_path, method, mask_name, video, os.path.basename(f))
                                     for f in clip]
                        else:
                            masks = [os.path.join(video_path, method, mask_name, video, 'mask_' + os.path.basename(f))
                                     for f in clip]
                        dataset_samples.append({
                            'type': 'video', 'dataset_type': datatype, 'dataset_path': video_path,
                            'mode': 'clip', 'clip_paths': clip, 'mask_paths': masks, 'method': method,
                        })
        else:
            for method in video_methods:
                fake_dir = os.path.join(video_path, 'fake', method)
                if not os.path.isdir(fake_dir):
                    continue
                for video in sorted(os.listdir(fake_dir)):
                    frame_dir = os.path.join(fake_dir, video)
                    if not os.path.isdir(frame_dir):
                        continue
                    frames = sorted(glob(os.path.join(frame_dir, '*.jpg')))
                    if len(frames) < self.clip_len:
                        continue

                    if self.split == 'test':
                        starts = list(range(min(10, len(frames) - self.clip_len + 1)))
                    else:
                        starts = list(range(0, max(0, len(frames) - self.clip_len), self.clip_len))[:2]

                    for i in starts:
                        clip = frames[i:i + self.clip_len]
                        if len(clip) != self.clip_len:
                            continue
                        masks = [os.path.join(video_path, 'mask', video, os.path.basename(f)) for f in clip]
                        dataset_samples.append({
                            'type': 'video', 'dataset_type': 'Other', 'dataset_path': video_path,
                            'mode': 'clip', 'clip_paths': clip, 'mask_paths': masks, 'method': method,
                        })

    # -------------------------------------------------------------- transforms
    def _setup_transforms(self, image_transforms, video_transforms):
        # Final padding / resizing transform (shared layout for images).
        if self.is_padding:
            self.post_transform = get_albu_transforms(type_="pad", output_size=self.output_size)
        else:
            self.post_transform = get_albu_transforms(type_="resize", output_size=self.output_size)

        self.image_transforms = image_transforms

        # Video resize/pad applied before normalization.
        if self.is_padding:
            self.video_before_trans = A.Compose([
                A.PadIfNeeded(min_height=self.output_size[0], min_width=self.output_size[1],
                              border_mode=0, value=0, position='top_left', mask_value=0),
            ])
        else:
            self.video_before_trans = A.Compose([A.Resize(self.output_size[0], self.output_size[1])])

        if video_transforms is None:
            if self.is_padding:
                base_transforms = [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    A.Crop(0, 0, self.output_size[0], self.output_size[1]),
                    ToTensorV2(transpose_mask=True),
                ]
            else:
                base_transforms = [
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(transpose_mask=True),
                ]

            if self.split == 'train':
                train_transforms = [
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit=30, p=0.7, border_mode=0),
                ]
                self.video_data_transform = A.ReplayCompose(train_transforms)
            else:
                self.video_data_transform = None

            self.video_base_transform = A.Compose(base_transforms)
        else:
            self.video_data_transform = None
            self.video_base_transform = video_transforms

    # ----------------------------------------------------------- item builders
    def _process_image_as_video(self, sample: Dict) -> Dict[str, Any]:
        """Load an image sample. Returns plain (C, H, W) tensors; the collate_fn
        promotes a single-frame batch via torch.stack."""
        tp_path = sample['tp_path']
        gt_path = sample['gt_path']

        tp_img = self.img_loader(tp_path)
        tp_shape = tp_img.size
        tp_img = np.array(tp_img)  # H W C

        if gt_path != "Negative":
            gt_img = self.img_loader(gt_path)
            gt_shape = gt_img.size
            gt_img = np.array(gt_img)
            label = 1
        else:
            gt_img = np.zeros((tp_img.shape[0], tp_img.shape[1], 3))
            gt_shape = (tp_img.shape[1], tp_img.shape[0])
            label = 0

        assert tp_shape == gt_shape, (
            f"tp and gt image shape must be the same, but got {tp_shape} and {gt_shape} "
            f"for '{tp_path}' and '{gt_path}'. Please check it!")

        if self.image_transforms is not None:
            res_dict = self.image_transforms(image=tp_img, mask=gt_img)
            tp_img = res_dict['image']
            gt_img = res_dict['mask']
            label = 0 if np.all(gt_img == 0) else 1

        tp_shape = tp_img.shape[0:2]  # H, W

        gt_img = (np.mean(gt_img, axis=2, keepdims=True) > 127.5) * 1.0
        gt_img = gt_img.transpose(2, 0, 1)[0]  # H W
        masks_list = [gt_img]

        if self.edge_mask_generator is not None:
            gt_img_edge = self.edge_mask_generator(torch.from_numpy(gt_img).float())[0]
            masks_list.append(gt_img_edge.numpy())

        res_dict = self.post_transform(image=tp_img, masks=masks_list)
        tp_img = res_dict['image']
        gt_img = res_dict['masks'][0].unsqueeze(0)  # 1 H W

        final_shape = self.output_size if self.is_resizing else tp_shape

        # Plain per-frame tensors; the collate_fn will stack or cat depending on layout.
        data_dict = {
            'image': tp_img.to(torch.float),           # C H W
            'mask':  gt_img,                            # 1 H W
            'label': torch.tensor(label),               # scalar
            'clip_len': torch.tensor(1),                # scalar; image == 1-frame clip
            'origin_shape': torch.tensor(tp_shape),     # (2,)
            'shape': torch.tensor(final_shape),
            'name': os.path.basename(tp_path),
            '_T': 1,                                    # marker used by collate_fn
        }

        if self.edge_mask_generator is not None:
            data_dict['edge_mask'] = res_dict['masks'][1].unsqueeze(0)  # 1 H W

        if self.is_padding:
            shape_mask = torch.zeros_like(gt_img)
            shape_mask[:, :tp_shape[0], :tp_shape[1]] = 1
            data_dict['shape_mask'] = shape_mask        # 1 H W

        return data_dict

    def _process_video(self, sample: Dict) -> Dict[str, Any]:
        """Load a video clip sample (T = clip_len) with frame-consistent augmentation."""
        clip_paths = sample['clip_paths']
        mask_paths = sample['mask_paths']

        imgs = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in clip_paths]
        ms = []
        for i, m in enumerate(mask_paths):
            if os.path.exists(m):
                mask = (cv2.imread(m, cv2.IMREAD_GRAYSCALE) > 127.5).astype(np.float32)
                h_img, w_img = imgs[i].shape[:2]
                if (h_img, w_img) != mask.shape[:2]:
                    mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                ms.append(mask)
            else:
                h, w = imgs[i].shape[:2]
                ms.append(np.zeros((h, w), dtype=np.float32))

        tp_shape = imgs[0].shape[:2] if self.is_padding else self.output_size

        original_edge_masks = [np.zeros_like(mask) for mask in ms] if self.edge_mask_generator is not None else None

        # Step 1: frame-consistent data augmentation (ReplayCompose).
        if self.video_data_transform is not None:
            aug = self.video_data_transform(image=imgs[0], mask=ms[0])
            replay = aug['replay']
            data_aug_imgs, data_aug_masks, data_aug_edges = [], [], []
            for i, (img, mask) in enumerate(zip(imgs, ms)):
                aug_result = A.ReplayCompose.replay(replay, image=img, mask=mask)
                data_aug_imgs.append(aug_result['image'])
                data_aug_masks.append(aug_result['mask'])
                if self.edge_mask_generator is not None:
                    edge_result = A.ReplayCompose.replay(replay, image=img, mask=original_edge_masks[i])
                    data_aug_edges.append(edge_result['mask'])
        else:
            data_aug_imgs, data_aug_masks = imgs, ms
            data_aug_edges = original_edge_masks if self.edge_mask_generator is not None else None

        # Step 2: before transform (resize / pad).
        before_trans_imgs = [self.video_before_trans(image=img)['image'] for img in data_aug_imgs]
        before_trans_masks = [self.video_before_trans(image=mask)['image'] for mask in data_aug_masks]
        if self.edge_mask_generator is not None:
            before_trans_edges = [self.video_before_trans(image=edge)['image'] for edge in data_aug_edges]

        # Step 3: base transform (normalize + to-tensor).
        transformed_imgs, transformed_masks, transformed_edges = [], [], []
        for i, (img, mask) in enumerate(zip(before_trans_imgs, before_trans_masks)):
            base_result = self.video_base_transform(image=img, mask=mask)
            transformed_imgs.append(base_result['image'])
            transformed_masks.append(base_result['mask'])
            if self.edge_mask_generator is not None:
                base_edge = self.video_base_transform(image=img, mask=before_trans_edges[i])['mask']
                transformed_edges.append(base_edge.float().unsqueeze(0))

        mask_tensors = []
        for mask in transformed_masks:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            mask = mask.float()
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask_tensors.append(mask)

        labels = [0 if torch.sum(mask) == 0 else 1 for mask in mask_tensors]
        video_name = (os.path.basename(os.path.dirname(clip_paths[0]))
                      + os.path.basename(clip_paths[0]).split('.')[0])

        # Per-frame tensors live in (T, *) layout; collate_fn handles per-sample T.
        data_dict = {
            'image': torch.stack(transformed_imgs, dim=0),   # T C H W
            'mask': torch.stack(mask_tensors, dim=0),        # T 1 H W
            'label': torch.tensor(labels[0]),                # representative scalar
            'clip_len': torch.tensor(self.clip_len),
            'origin_shape': torch.tensor(tp_shape),
            'shape': torch.tensor(tp_shape),
            'name': video_name,
            '_T': self.clip_len,
        }

        if self.edge_mask_generator is not None:
            data_dict['edge_mask'] = torch.stack(transformed_edges, dim=0)  # T 1 H W

        if self.is_padding:
            shape_mask = torch.zeros_like(mask_tensors[0])
            shape_mask[:, :tp_shape[0], :tp_shape[1]] = 1
            T = len(transformed_imgs)
            data_dict['shape_mask'] = shape_mask.unsqueeze(0).repeat(T, 1, 1, 1)

        return data_dict

    # --------------------------------------------------------------- sampling
    def _setup_sampling(self):
        num_image_datasets = len(self.image_datasets_samples)
        num_video_datasets = len(self.video_datasets_samples)
        if num_image_datasets == 0 and num_video_datasets == 0:
            raise ValueError("No datasets found")

        total_image_samples = sum(len(s) for s in self.image_datasets_samples)
        total_video_samples = sum(len(s) for s in self.video_datasets_samples)
        self._len = total_image_samples + total_video_samples

        if self.split == 'train':
            # Flatten (type, dataset_idx, sample_idx). The DataLoader sampler owns shuffling,
            # matching IMDLBenCo's dataset behavior and keeping DDP epochs reproducible.
            self.all_samples_indices = []
            for dataset_idx, dataset_samples in enumerate(self.image_datasets_samples):
                for i in range(len(dataset_samples)):
                    self.all_samples_indices.append(('image', dataset_idx, i))
            for dataset_idx, dataset_samples in enumerate(self.video_datasets_samples):
                for i in range(len(dataset_samples)):
                    self.all_samples_indices.append(('video', dataset_idx, i))
            print("Training mode sampling setup:")
            print(f"  Image datasets: {num_image_datasets}, total samples: {total_image_samples}")
            print(f"  Video datasets: {num_video_datasets}, total samples: {total_video_samples}")
            print(f"  Total samples per epoch: {self._len}")
        else:
            print(f"Test mode: using all {self._len} samples")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.split == 'train':
            sample_type, dataset_idx, sample_idx = self.all_samples_indices[index]
            if sample_type == 'image':
                sample = self.image_datasets_samples[dataset_idx][sample_idx]
            else:
                sample = self.video_datasets_samples[dataset_idx][sample_idx]
        else:
            current_index = 0
            sample = None
            for dataset_samples in self.image_datasets_samples:
                if index < current_index + len(dataset_samples):
                    sample = dataset_samples[index - current_index]
                    break
                current_index += len(dataset_samples)
            if sample is None:
                for dataset_samples in self.video_datasets_samples:
                    if index < current_index + len(dataset_samples):
                        sample = dataset_samples[index - current_index]
                        break
                    current_index += len(dataset_samples)
            if sample is None:
                raise IndexError(f"Index {index} out of range")

        if sample['type'] == 'image':
            data_dict = self._process_image_as_video(sample)
        else:
            data_dict = self._process_video(sample)

        if self.post_funcs is not None:
            if isinstance(self.post_funcs, list):
                for func in self.post_funcs:
                    if callable(func):
                        func(data_dict)
                    else:
                        raise NotImplementedError(f"Element {func} in list is not callable")
            elif callable(self.post_funcs):
                self.post_funcs(data_dict)
            else:
                raise NotImplementedError(f"Unsupported type: {type(self.post_funcs)}")

        return data_dict

    @staticmethod
    def collate_fn(batch):
        """
        Flatten variable-length clips (image -> T=1, video -> T=clip_len) into a single
        frame batch.  Per-sample items returned by `_process_image_as_video` are 3D
        ((C,H,W) / (1,H,W)); video items are 4D ((T,C,H,W)).

        Fast path: when every sample is a single image, use torch.stack (avoids the
        ~36% extra cost of cat-with-3D-tensors via Python list iteration).  Mixed or
        pure-video batches fall back to the general per-sample unsqueeze+cat path.

        Returns a dict with:
          - image        : (sum_T, C, H, W) flattened frames
          - mask         : (sum_T, 1, H, W)
          - clip_len     : (B,) frames-per-sample
          - origin_shape : (B, 2)
          - label        : (B,) representative (first-frame) label per sample
        """
        first = batch[0]
        all_image = all(s.get('_T', 1) == 1 for s in batch)

        if all_image:
            # 3D per-sample tensors -> single stack call per modality.
            image = torch.stack([s['image'] for s in batch], dim=0)
            mask  = torch.stack([s['mask']  for s in batch], dim=0)
            result = {
                'image': image,
                'mask':  mask,
                'label': torch.stack([s['label'] for s in batch]),
                'clip_len': torch.stack([s['clip_len'] for s in batch]),
                'origin_shape': torch.stack([s['origin_shape'] for s in batch]),
            }
            if 'edge_mask' in first:
                result['edge_mask'] = torch.stack([s['edge_mask'] for s in batch], dim=0)
            if 'shape_mask' in first:
                result['shape_mask'] = torch.stack([s['shape_mask'] for s in batch], dim=0)
            return result

        # Mixed image+video batch: promote 3D samples to 4D (T=1) then cat along T.
        def _as_T(t):
            return t.unsqueeze(0) if t.dim() == 3 else t

        image_cat = torch.cat([_as_T(s['image']) for s in batch], dim=0)
        mask_cat  = torch.cat([_as_T(s['mask'])  for s in batch], dim=0)
        result = {
            'image': image_cat,
            'mask':  mask_cat,
            'label': torch.stack([s['label'] for s in batch]),
            'clip_len': torch.stack([s['clip_len'] for s in batch]),
            'origin_shape': torch.stack([s['origin_shape'] for s in batch]),
        }
        if 'edge_mask' in first:
            result['edge_mask'] = torch.cat([_as_T(s['edge_mask']) for s in batch], dim=0)
        if 'shape_mask' in first:
            result['shape_mask'] = torch.cat([_as_T(s['shape_mask']) for s in batch], dim=0)
        return result

    def __len__(self) -> int:
        return self._len

    def __str__(self) -> str:
        total_image_samples = sum(len(s) for s in self.image_datasets_samples)
        total_video_samples = sum(len(s) for s in self.video_datasets_samples)
        return (f"[MixedImageVideoDataset] Image datasets: {len(self.image_datasets_samples)} "
                f"(total: {total_image_samples}), Video datasets: {len(self.video_datasets_samples)} "
                f"(total: {total_video_samples}), Epoch length: {self._len}")


def split_dataset_config(config: List[List]) -> Tuple[List[List], List[List]]:
    """Split a flat dataset-config list into (image_datasets, video_datasets)."""
    image_datasets, video_datasets = [], []
    for entry in config:
        dataset_type = entry[0]
        if dataset_type in ('ManiDataset', 'JsonDataset'):
            image_datasets.append(entry)
        elif dataset_type == 'VideoDataset':
            video_datasets.append(entry)
        else:
            # Unknown type -> treat as image dataset.
            image_datasets.append(entry)
    return image_datasets, video_datasets


def build_dataset_from_config(config, split: str = 'train', **kwargs) -> MixedImageVideoDataset:
    """
    Build a MixedImageVideoDataset from a config (path to JSON file or already-parsed list).

    The same helper produces image-only, video-only, or mixed datasets depending on the
    entries present in `config`.
    """
    if isinstance(config, str):
        with open(config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    image_datasets, video_datasets = split_dataset_config(config)
    return MixedImageVideoDataset(
        image_datasets=image_datasets if image_datasets else None,
        video_datasets=video_datasets if video_datasets else None,
        split=split,
        **kwargs,
    )
