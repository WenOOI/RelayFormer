import os
import cv2
import torch
import numpy as np
import shutil
from PIL import Image
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import argparse

from inference_dataset import create_inference_dataloader
from model.RelayFormer import RelayFormer

class InferenceProcessor:
    """
    Inference processor for handling images and videos, outputting prediction results
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 output_dir: str,
                 device: str = 'cuda',
                 overlay_alpha: float = 0.3,
                 mask_threshold: float = 0.5):
        """
        Args:
            model: Pretrained model that takes image as input and returns mask
            output_dir: Output directory
            device: Device ('cuda' or 'cpu')
            overlay_alpha: Overlay transparency
            mask_threshold: Threshold for binarizing mask
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = output_dir
        self.overlay_alpha = overlay_alpha
        self.mask_threshold = mask_threshold
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.detach().numpy()
    
    def _denormalize_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Denormalize image"""
        # ImageNet normalization parameters
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img = self._tensor_to_numpy(img_tensor)
        
        if img.ndim == 4:  # (B, C, H, W)
            img = img[0]  # Take first batch
        
        if img.ndim == 3:  # (C, H, W)
            img = img.transpose(1, 2, 0)  # -> (H, W, C)
        
        # Denormalize
        img = img * std + mean
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        return img
    
    def _process_mask(self, mask_tensor: torch.Tensor) -> np.ndarray:
        """Process mask tensor"""
        mask = self._tensor_to_numpy(mask_tensor)
        
        if mask.ndim == 4:  # (B, C, H, W)
            mask = mask[0]  # Take first batch
        
        if mask.ndim == 3:  # (C, H, W)
            mask = mask[0]  # Take first channel
        
        # Binarize
        mask = (mask > self.mask_threshold).astype(np.uint8) * 255
        
        return mask
    
    def _create_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create mask overlay"""
        overlay = image.copy()
        
        # Create red overlay
        red_overlay = np.zeros_like(image)
        red_overlay[:, :, 2] = 255  # Red channel
        
        # Apply mask
        mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
        overlay = overlay.astype(np.float32)
        red_overlay = red_overlay.astype(np.float32)
        
        # Blend
        overlay = overlay * (1 - mask_3ch * self.overlay_alpha) + red_overlay * mask_3ch * self.overlay_alpha
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        return overlay
    
    def _resize_mask_to_original(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Resize mask back to original dimensions"""
        if mask.shape != original_shape:
            mask = cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    
    def _resize_image_to_original(self, image: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Resize image back to original dimensions"""
        if image.shape[:2] != original_shape:
            image = cv2.resize(image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        return image
    
    def _crop_mask_to_original(self, mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Crop mask back to original dimensions (for top-left padding cases)"""
        h, w = original_shape
        return mask[:h, :w]  # Take top-left original dimension area

    def _crop_image_to_original(self, image: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        """Crop image back to original dimensions (for top-left padding cases)"""
        h, w = original_shape
        return image[:h, :w, :]  # Keep all color channels
    
    def _copy_original_file(self, sample_name: str, sample_type: str, sample_dir: str, batch: Dict[str, Any]):
        """Copy original file to output directory"""
        # Need to get original file path based on actual situation
        # Since InferenceDataset doesn't save original path, provide an example implementation
        
        if sample_type == 'image':
            # For images, we save processed original
            original_file = os.path.join(sample_dir, f"original_{sample_name}")
            # Should get original image from dataset, skip for now
            pass
        else:
            # For videos, create copy of original video
            original_file = os.path.join(sample_dir, f"original_{sample_name}")
            # Should get original video from dataset, skip for now
            pass
    
    def _save_image_results(self, sample_name: str, sample_dir: str, 
                           image: np.ndarray, mask: np.ndarray, overlay: np.ndarray):
        """Save image results"""
        # Remove file extension to get base name
        base_name = os.path.splitext(sample_name)[0]
        
        # Save original image (processed)
        Image.fromarray(image).save(os.path.join(sample_dir, f"{base_name}_original.png"))
        
        # Save mask
        Image.fromarray(mask).save(os.path.join(sample_dir, f"{base_name}_mask.png"))
        
        # Save overlay
        Image.fromarray(overlay).save(os.path.join(sample_dir, f"{base_name}_overlay.png"))
    
    def _save_video_results(self, sample_name: str, sample_dir: str,
                           frames: list, masks: list, overlays: list):
        """Save video results"""
        base_name = os.path.splitext(sample_name)[0]
        
        # Save frames
        frames_dir = os.path.join(sample_dir, "frames")
        masks_dir = os.path.join(sample_dir, "masks") 
        overlays_dir = os.path.join(sample_dir, "overlays")
        
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)
        
        for i, (frame, mask, overlay) in enumerate(zip(frames, masks, overlays)):
            # Save frame
            Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            Image.fromarray(mask).save(os.path.join(masks_dir, f"mask_{i:04d}.png"))
            Image.fromarray(overlay).save(os.path.join(overlays_dir, f"overlay_{i:04d}.png"))
        
        # Create video files
        self._create_video_from_frames(frames, os.path.join(sample_dir, f"{base_name}_original.mp4"))
        self._create_video_from_frames(overlays, os.path.join(sample_dir, f"{base_name}_overlay.mp4"))
        
        # Create mask video (convert to 3 channels)
        mask_frames = [cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) for mask in masks]
        self._create_video_from_frames(mask_frames, os.path.join(sample_dir, f"{base_name}_mask.mp4"))
    
    def _create_video_from_frames(self, frames: list, output_path: str, fps: int = 30):
        """Create video from frame sequence"""
        if not frames:
            return
            
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame in frames:
                if len(frame.shape) == 3:
                    # RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                writer.write(frame_bgr)
        finally:
            writer.release()
    
    def process_batch(self, batch: Dict[str, Any], sample_type) -> Dict[str, Any]:
        """Process single batch"""
        with torch.no_grad():
            if 'image' in sample_type:
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                output = self.model(**batch)
            else:
                batch_size = batch['image'].shape[0]
                output = []
                for batch_idx in range(batch_size):
                    sample = {}
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            if batch[key].ndim == 5: sample[key] = batch[key][batch_idx].to(self.device)
                            else: sample[key] = batch[key].to(self.device)
                    i_output = self.model(**sample)
                    output.append(i_output)
                output = torch.stack(output)
            return {
                'mask': output,
                'input': batch['image']
            }
    
    def process_single_sample(self, batch: Dict[str, Any]):
        """Process single sample"""
        sample_name = batch['name'][0]
        sample_type = batch['type'][0]
        origin_shape = tuple(batch['origin_shape'].tolist())[0]
        
        print(f"Processing {sample_type}: {sample_name}")
        
        # Create sample output directory
        base_name = os.path.splitext(sample_name)[0]
        sample_dir = os.path.join(self.output_dir, base_name)
        os.makedirs(sample_dir, exist_ok=True)

        # Model inference
        results = self.process_batch(batch, sample_type)
        
        input_tensor = results['input']
        mask_tensor = results['mask']
        
        if sample_type == 'image':
            # Process image
            image = self._denormalize_image(input_tensor)
            mask = self._process_mask(mask_tensor)
            
            # Resize to original dimensions
            image = self._crop_image_to_original(image, origin_shape)
            mask = self._crop_mask_to_original(mask, origin_shape)
            
            # Create overlay
            overlay = self._create_overlay(image, mask)
            
            # Save results
            self._save_image_results(sample_name, sample_dir, image, mask, overlay)
            
        else:
            # Process video
            batch_size, clip_len = input_tensor.shape[0], input_tensor.shape[1] if input_tensor.ndim == 5 else 1
            
            if input_tensor.ndim == 4:  # Single frame processed as video
                input_tensor = input_tensor.unsqueeze(1)  # (B, 1, C, H, W)
                mask_tensor = mask_tensor.unsqueeze(1) if mask_tensor.ndim == 4 else mask_tensor

            frames = []
            masks = []
            overlays = []
            B, T, C, H, W = input_tensor.shape
            # input_tensor = input_tensor.view(-1, C, H, W)
            
            for b in range(B): 
                for t in range(T):
                    frame = self._denormalize_image(input_tensor[b:b+1, t])
                    mask = self._process_mask(mask_tensor[b:b+1, t])
                    
                    # Resize to original dimensions
                    frame = self._crop_image_to_original(frame, origin_shape)
                    mask = self._crop_mask_to_original(mask, origin_shape)
                    
                    # Create overlay
                    overlay = self._create_overlay(frame, mask)
                    
                    frames.append(frame)
                    masks.append(mask)
                    overlays.append(overlay)
            
            # Save video results
            self._save_video_results(sample_name, sample_dir, frames, masks, overlays)
    
    def run(self, data_path: str, **dataloader_kwargs):
        """Run inference processing"""
        # Create data loader
        dataloader = create_inference_dataloader(
            data_path=data_path,
            batch_size=1,  # batch_size=1 during inference
            **dataloader_kwargs
        )
        
        print(f"Processing {len(dataloader)} samples...")
        print(f"Output directory: {self.output_dir}")
        
        # Process each sample
        for batch in tqdm(dataloader, desc="Processing"):
            # try:
            self.process_single_sample(batch)
            # except Exception as e:
            #     print(f"Error processing {batch.get('name', 'unknown')}: {str(e)}")
            #     continue
        
        print("Processing completed!")


def create_model(model_path):
    model = RelayFormer()
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])     
    return model


def main():
    parser = argparse.ArgumentParser(description='Run inference on images and videos')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory path')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    parser.add_argument('--model_path', type=str, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--output_size', nargs=2, type=int, default=[1024, 1024], 
                       help='Output image size (height width)')
    parser.add_argument('--clip_len', type=int, default=5, help='Video clip length')
    parser.add_argument('--overlay_alpha', type=float, default=0.3, help='Overlay transparency')
    parser.add_argument('--mask_threshold', type=float, default=0.5, help='Mask threshold')
    
    args = parser.parse_args()
    
    
    model = create_model(args.model_path)
    
    # Create processor
    processor = InferenceProcessor(
        model=model,
        output_dir=args.output_dir,
        device=args.device,
        overlay_alpha=args.overlay_alpha,
        mask_threshold=args.mask_threshold
    )
    
    # Run processing
    processor.run(
        data_path=args.input_dir,
        output_size=tuple(args.output_size),
        clip_len=args.clip_len,
        normalize=True
    )


if __name__ == "__main__":
    main()
