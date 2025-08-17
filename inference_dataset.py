import os
import cv2
import torch
import numpy as np
from glob import glob
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any, Union
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def pil_loader(path: str) -> Image.Image:
    """PIL image loader"""
    return Image.open(path).convert('RGB')


def get_basic_transforms(output_size: Tuple[int, int], normalize: bool = True):
    """Get basic transforms for inference"""
    transforms_list = [
        A.PadIfNeeded(          
                    min_height=output_size[0],
                    min_width=output_size[1], 
                    border_mode=0, 
                    value=0, 
                    position='top_left',
                    mask_value=0),
        A.Crop(0, 0, output_size[0], output_size[1]),
    ]
    
    if normalize:
        transforms_list.append(
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    transforms_list.append(ToTensorV2())
    
    return A.Compose(transforms_list)


class InferenceDataset(Dataset):
    """
    Simplified inference dataset for loading images or videos for inference
    Supports reading images and videos from folders with basic preprocessing
    """
    
    def __init__(self, 
                 data_path: str,
                 output_size: Tuple[int, int] = (1024, 1024),
                 clip_len: int = 5,
                 normalize: bool = True,
                 img_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp'),
                 video_extensions: Tuple[str, ...] = ('.mp4', '.avi', '.mov', '.mkv'),
                 img_loader = pil_loader) -> None:
        """
        Args:
            data_path: Data path, can be a folder containing images/videos
            output_size: Output image size (height, width)
            clip_len: Video clip length
            normalize: Whether to apply ImageNet normalization
            img_extensions: Supported image formats
            video_extensions: Supported video formats
            img_loader: Image loader
        """
        
        self.data_path = data_path
        self.output_size = output_size
        self.clip_len = clip_len
        self.img_loader = img_loader
        
        # Initialize transforms
        self.transforms = get_basic_transforms(output_size, normalize)
        
        # Scan data
        self.samples = []
        self._scan_data(data_path, img_extensions, video_extensions)
        
        print(f"InferenceDataset initialized with {len(self.samples)} samples")
    
    def _scan_data(self, data_path: str, img_extensions: Tuple[str, ...], video_extensions: Tuple[str, ...]):
        """Scan data path to collect image and video files"""
        
        if os.path.isfile(data_path):
            # Single file
            if data_path.lower().endswith(img_extensions):
                self.samples.append({
                    'type': 'image',
                    'path': data_path,
                    'name': os.path.basename(data_path)
                })
            elif data_path.lower().endswith(video_extensions):
                self.samples.append({
                    'type': 'video_file',
                    'path': data_path,
                    'name': os.path.basename(data_path)
                })
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
                
        elif os.path.isdir(data_path):
            # Directory
            self._scan_directory(data_path, img_extensions, video_extensions)
        else:
            raise ValueError(f"Path does not exist: {data_path}")
    
    def _scan_directory(self, dir_path: str, img_extensions: Tuple[str, ...], video_extensions: Tuple[str, ...]):
        """Scan files in directory"""
        
        # Check if contains video frame sequences (frames organized in subfolders)
        # subdirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        
        # if subdirs:
        #     # Check if subfolders contain video frames
        #     for subdir in sorted(subdirs):
        #         subdir_path = os.path.join(dir_path, subdir)
        #         frame_files = []
                
        #         for ext in img_extensions:
        #             frame_files.extend(glob(os.path.join(subdir_path, f'*{ext}')))
                
        #         if len(frame_files) >= self.clip_len:
        #             # Video frame sequence
        #             frame_files.sort()
        #             self.samples.append({
        #                 'type': 'video_frames',
        #                 'frames': frame_files,
        #                 'name': subdir
        #             })
        
        # Scan direct image and video files
        for file_name in sorted(os.listdir(dir_path)):
            file_path = os.path.join(dir_path, file_name)
            
            if os.path.isfile(file_path):
                if file_name.lower().endswith(img_extensions):
                    self.samples.append({
                        'type': 'image',
                        'path': file_path,
                        'name': file_name
                    })
                elif file_name.lower().endswith(video_extensions):
                    self.samples.append({
                        'type': 'video_file',
                        'path': file_path,
                        'name': file_name
                    })
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load single image"""
        img = self.img_loader(image_path)
        return np.array(img)
    
    def _load_video_from_file(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        finally:
            cap.release()
        
        return frames
    
    def _load_video_from_frames(self, frame_paths: List[str]) -> List[np.ndarray]:
        """Load video from frame sequence"""
        frames = []
        for frame_path in frame_paths:
            if frame_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.imread(frame_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(img)
        return frames
    
    def _process_image(self, sample: Dict) -> Dict[str, Any]:
        """Process single image"""
        image_path = sample['path']
        image_name = sample['name']
        
        # Load image
        img = self._load_image(image_path)
        origin_shape = img.shape[:2]  # (H, W)
        
        # Apply transforms
        transformed = self.transforms(image=img)
        img_tensor = transformed['image']  # (C, H, W)
        
        return {
            'image': img_tensor.unsqueeze(0),  # (1, C, H, W) - as single-frame "video"
            'clip': img_tensor.unsqueeze(0),
            'clip_len': torch.tensor(1),
            'origin_shape': torch.tensor(origin_shape),
            'final_shape': torch.tensor(self.output_size),
            'name': image_name,
            'type': 'image'
        }
    
    # def _process_video(self, sample: Dict) -> Dict[str, Any]:
    #     """Process video"""
    #     video_name = sample['name']
        
    #     # Load video frames
    #     if sample['type'] == 'video_file':
    #         frames = self._load_video_from_file(sample['path'])
    #     else:  # video_frames
    #         frames = self._load_video_from_frames(sample['frames'])
        
    #     if len(frames) == 0:
    #         raise ValueError(f"No frames loaded from {video_name}")
        
    #     # Ensure enough frames
    #     if len(frames) < self.clip_len:
    #         # If not enough frames, repeat last frame
    #         last_frame = frames[-1]
    #         while len(frames) < self.clip_len:
    #             frames.append(last_frame.copy())
        
    #     # Select consecutive frames
    #     if len(frames) > self.clip_len:
    #         # Take clip_len frames from start, or can be changed to take from middle
    #         frames = frames[:self.clip_len]
        
    #     origin_shape = frames[0].shape[:2]  # (H, W)
        
    #     # Apply transforms to each frame
    #     transformed_frames = []
    #     for frame in frames:
    #         transformed = self.transforms(image=frame)
    #         transformed_frames.append(transformed['image'])
        
    #     # Stack all frames
    #     video_tensor = torch.stack(transformed_frames, dim=0)  # (T, C, H, W)
        
    #     return {
    #         'image': video_tensor,  # (T, C, H, W)
    #         'clip': video_tensor,
    #         'clip_len': torch.tensor(len(frames)),
    #         'origin_shape': torch.tensor(origin_shape),
    #         'final_shape': torch.tensor(self.output_size),
    #         'name': video_name,
    #         'type': 'video'
    #     }
    def _process_video(self, sample: Dict) -> Dict[str, Any]:
        """Process video"""
        video_name = sample['name']
        
        # Load video frames
        if sample['type'] == 'video_file':
            frames = self._load_video_from_file(sample['path'])
        else:  # video_frames
            frames = self._load_video_from_frames(sample['frames'])
        
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {video_name}")
        
        # Ensure enough frames
        if len(frames) < self.clip_len:
            # If not enough frames, repeat last frame
            last_frame = frames[-1]
            while len(frames) < self.clip_len:
                frames.append(last_frame.copy())
        
        origin_shape = frames[0].shape[:2]  # (H, W)
        
        # Apply transforms to each frame
        transformed_frames = []
        for frame in frames:
            transformed = self.transforms(image=frame)
            transformed_frames.append(transformed['image'])
        
        # Process frames to create all possible clips
        video_tensors = []
        if len(frames) <= self.clip_len:
            # If frame count <= clip_len, stack directly
            video_tensor = torch.stack(transformed_frames, dim=0)  # (T, C, H, W)
            video_tensors.append(video_tensor)
        else:
            # When frame count > clip_len, create all possible clips
            for i in range(0, len(frames) - self.clip_len + 1, self.clip_len):
                clip_frames = transformed_frames[i:i + self.clip_len]
                video_tensor = torch.stack(clip_frames, dim=0)  # (T, C, H, W)
                video_tensors.append(video_tensor)
        
        # Stack all clips into batch dimension
        final_video_tensor = torch.stack(video_tensors, dim=0)  # (B, T, C, H, W)
        
        return {
            'image': final_video_tensor,  # (B, T, C, H, W)
            'clip': final_video_tensor,
            'clip_len': torch.tensor(self.clip_len),
            'origin_shape': torch.tensor(origin_shape),
            'final_shape': torch.tensor(self.output_size),
            'name': video_name,
            'type': 'video'
        }
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get data sample"""
        sample = self.samples[index]
        
        if sample['type'] == 'image':
            return self._process_image(sample)
        else:  # video_file or video_frames
            return self._process_video(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __str__(self) -> str:
        image_count = sum(1 for s in self.samples if s['type'] == 'image')
        video_count = len(self.samples) - image_count
        return f"InferenceDataset: {image_count} images, {video_count} videos, total: {len(self.samples)}"


def create_inference_dataloader(data_path: str,
                              output_size: Tuple[int, int] = (1024, 1024),
                              clip_len: int = 5,
                              batch_size: int = 1,
                              normalize: bool = True,
                              num_workers: int = 0) -> torch.utils.data.DataLoader:
    """
    Create inference data loader
    
    Args:
        data_path: Data path
        output_size: Output size
        clip_len: Video clip length  
        batch_size: Batch size (usually 1 during inference)
        normalize: Whether to normalize
        num_workers: Number of worker processes
    
    Returns:
        DataLoader object
    """
    dataset = InferenceDataset(
        data_path=data_path,
        output_size=output_size,
        clip_len=clip_len,
        normalize=normalize
    )
    
    def collate_fn(batch):
        """Simple collate function to maintain batch structure"""
        # if len(batch) == 1:
        #     return batch[0]
        # else:
        #     # If batch_size > 1, need special handling
        return {
            'image': torch.cat([item['image'] for item in batch], dim=0),
            'clip': torch.cat([item['image'] for item in batch], dim=0),
            'clip_len': torch.stack([item['clip_len'] for item in batch]),
            'origin_shape': torch.stack([item['origin_shape'] for item in batch]),
            'final_shape': torch.stack([item['final_shape'] for item in batch]),
            'name': [item['name'] for item in batch],
            'type': [item['type'] for item in batch]
        }
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

