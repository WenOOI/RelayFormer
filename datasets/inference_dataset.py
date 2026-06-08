"""
Inference dataset for RelayFormer.

Scans a file or directory and yields preprocessed images / videos ready for the model.
Images are treated as single-frame clips; video files are split into consecutive clips of
`clip_len` frames. Preprocessing mirrors training: top-left pad to `output_size` + ImageNet
normalization, so a trained checkpoint sees inputs in the same distribution.
"""
import os
from typing import List, Tuple, Dict, Any

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def pil_loader(path: str) -> Image.Image:
    return Image.open(path).convert('RGB')


def get_basic_transforms(output_size: Tuple[int, int], normalize: bool = True):
    """Top-left pad + crop to output_size, optional ImageNet normalization, to-tensor."""
    transforms_list = [
        A.PadIfNeeded(min_height=output_size[0], min_width=output_size[1],
                      border_mode=0, value=0, position='top_left', mask_value=0),
        A.Crop(0, 0, output_size[0], output_size[1]),
    ]
    if normalize:
        transforms_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transforms_list.append(ToTensorV2())
    return A.Compose(transforms_list)


class InferenceDataset(Dataset):
    """Load images and video files from a path for inference."""

    def __init__(self,
                 data_path: str,
                 output_size: Tuple[int, int] = (1024, 1024),
                 clip_len: int = 4,
                 normalize: bool = True,
                 img_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
                 video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv'),
                 img_loader=pil_loader) -> None:
        self.data_path = data_path
        self.output_size = output_size
        self.clip_len = clip_len
        self.img_loader = img_loader
        self.transforms = get_basic_transforms(output_size, normalize)

        self.samples: List[Dict] = []
        self._scan_data(data_path, img_extensions, video_extensions)
        print(f"InferenceDataset initialized with {len(self.samples)} samples")

    def _scan_data(self, data_path, img_extensions, video_extensions):
        if os.path.isfile(data_path):
            if data_path.lower().endswith(img_extensions):
                self.samples.append({'type': 'image', 'path': data_path, 'name': os.path.basename(data_path)})
            elif data_path.lower().endswith(video_extensions):
                self.samples.append({'type': 'video_file', 'path': data_path, 'name': os.path.basename(data_path)})
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
        elif os.path.isdir(data_path):
            for file_name in sorted(os.listdir(data_path)):
                file_path = os.path.join(data_path, file_name)
                if not os.path.isfile(file_path):
                    continue
                if file_name.lower().endswith(img_extensions):
                    self.samples.append({'type': 'image', 'path': file_path, 'name': file_name})
                elif file_name.lower().endswith(video_extensions):
                    self.samples.append({'type': 'video_file', 'path': file_path, 'name': file_name})
        else:
            raise ValueError(f"Path does not exist: {data_path}")

    def _load_image(self, image_path: str) -> np.ndarray:
        return np.array(self.img_loader(image_path))

    def _load_video_from_file(self, video_path: str) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return frames

    def _process_image(self, sample: Dict) -> Dict[str, Any]:
        img = self._load_image(sample['path'])
        origin_shape = img.shape[:2]  # (H, W)
        img_tensor = self.transforms(image=img)['image']  # (C, H, W)
        return {
            'image': img_tensor.unsqueeze(0),   # (1, C, H, W) single-frame clip
            'clip': img_tensor.unsqueeze(0),
            'clip_len': torch.tensor(1),
            'origin_shape': torch.tensor(origin_shape),
            'final_shape': torch.tensor(self.output_size),
            'name': sample['name'],
            'type': 'image',
        }

    def _process_video(self, sample: Dict) -> Dict[str, Any]:
        frames = self._load_video_from_file(sample['path'])
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {sample['name']}")

        # Pad with the last frame so the count is a clip_len multiple.
        if len(frames) < self.clip_len:
            while len(frames) < self.clip_len:
                frames.append(frames[-1].copy())

        origin_shape = frames[0].shape[:2]  # (H, W)
        transformed_frames = [self.transforms(image=f)['image'] for f in frames]

        # Split into consecutive non-overlapping clips of clip_len.
        video_tensors = []
        if len(transformed_frames) <= self.clip_len:
            video_tensors.append(torch.stack(transformed_frames, dim=0))
        else:
            for i in range(0, len(transformed_frames) - self.clip_len + 1, self.clip_len):
                video_tensors.append(torch.stack(transformed_frames[i:i + self.clip_len], dim=0))

        final_video_tensor = torch.stack(video_tensors, dim=0)  # (B, T, C, H, W)
        return {
            'image': final_video_tensor,
            'clip': final_video_tensor,
            'clip_len': torch.tensor(self.clip_len),
            'origin_shape': torch.tensor(origin_shape),
            'final_shape': torch.tensor(self.output_size),
            'name': sample['name'],
            'type': 'video',
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        if sample['type'] == 'image':
            return self._process_image(sample)
        return self._process_video(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        image_count = sum(1 for s in self.samples if s['type'] == 'image')
        video_count = len(self.samples) - image_count
        return f"InferenceDataset: {image_count} images, {video_count} videos, total: {len(self.samples)}"


def create_inference_dataloader(data_path: str,
                                output_size: Tuple[int, int] = (1024, 1024),
                                clip_len: int = 4,
                                batch_size: int = 1,
                                normalize: bool = True,
                                num_workers: int = 0) -> torch.utils.data.DataLoader:
    """Create an inference DataLoader (batch_size is typically 1)."""
    dataset = InferenceDataset(
        data_path=data_path, output_size=output_size, clip_len=clip_len, normalize=normalize)

    def collate_fn(batch):
        return {
            'image': torch.cat([item['image'] for item in batch], dim=0),
            'clip': torch.cat([item['image'] for item in batch], dim=0),
            'clip_len': torch.stack([item['clip_len'] for item in batch]),
            'origin_shape': torch.stack([item['origin_shape'] for item in batch]),
            'final_shape': torch.stack([item['final_shape'] for item in batch]),
            'name': [item['name'] for item in batch],
            'type': [item['type'] for item in batch],
        }

    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn)
