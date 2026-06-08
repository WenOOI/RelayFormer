"""
RelayFormer inference: localize manipulated regions in images and videos.

Loads a trained checkpoint, runs over a file or directory, and writes per-sample masks and
red overlays (and reassembled videos for video inputs).
"""
import os
import argparse
from typing import Dict, Any, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from models.RelayFormer import RelayFormer
from datasets.inference_dataset import create_inference_dataloader


class InferenceProcessor:
    """Run the model over a dataloader and save masks / overlays."""

    def __init__(self, model: torch.nn.Module, output_dir: str, device: str = 'cuda',
                 overlay_alpha: float = 0.3, mask_threshold: float = 0.5):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.output_dir = output_dir
        self.overlay_alpha = overlay_alpha
        self.mask_threshold = mask_threshold
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------- conversions
    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def _denormalize_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = self._tensor_to_numpy(img_tensor)
        if img.ndim == 4:
            img = img[0]
        if img.ndim == 3:
            img = img.transpose(1, 2, 0)
        img = img * std + mean
        return np.clip(img * 255, 0, 255).astype(np.uint8)

    def _process_mask(self, mask_tensor: torch.Tensor) -> np.ndarray:
        mask = self._tensor_to_numpy(mask_tensor)
        if mask.ndim == 4:
            mask = mask[0]
        if mask.ndim == 3:
            mask = mask[0]
        return (mask > self.mask_threshold).astype(np.uint8) * 255

    def _create_overlay(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        red_overlay = np.zeros_like(image)
        red_overlay[:, :, 2] = 255  # red channel (RGB)
        mask_3ch = np.stack([mask, mask, mask], axis=2) / 255.0
        overlay = image.astype(np.float32) * (1 - mask_3ch * self.overlay_alpha) \
            + red_overlay.astype(np.float32) * mask_3ch * self.overlay_alpha
        return np.clip(overlay, 0, 255).astype(np.uint8)

    @staticmethod
    def _crop_mask_to_original(mask: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        h, w = original_shape
        return mask[:h, :w]

    @staticmethod
    def _crop_image_to_original(image: np.ndarray, original_shape: Tuple[int, int]) -> np.ndarray:
        h, w = original_shape
        return image[:h, :w, :]

    # ------------------------------------------------------------------ saving
    @staticmethod
    def _save_image_results(sample_name, sample_dir, image, mask, overlay):
        base_name = os.path.splitext(sample_name)[0]
        Image.fromarray(image).save(os.path.join(sample_dir, f"{base_name}_original.png"))
        Image.fromarray(mask).save(os.path.join(sample_dir, f"{base_name}_mask.png"))
        Image.fromarray(overlay).save(os.path.join(sample_dir, f"{base_name}_overlay.png"))

    def _save_video_results(self, sample_name, sample_dir, frames, masks, overlays):
        base_name = os.path.splitext(sample_name)[0]
        frames_dir = os.path.join(sample_dir, "frames")
        masks_dir = os.path.join(sample_dir, "masks")
        overlays_dir = os.path.join(sample_dir, "overlays")
        for d in (frames_dir, masks_dir, overlays_dir):
            os.makedirs(d, exist_ok=True)

        for i, (frame, mask, overlay) in enumerate(zip(frames, masks, overlays)):
            Image.fromarray(frame).save(os.path.join(frames_dir, f"frame_{i:04d}.png"))
            Image.fromarray(mask).save(os.path.join(masks_dir, f"mask_{i:04d}.png"))
            Image.fromarray(overlay).save(os.path.join(overlays_dir, f"overlay_{i:04d}.png"))

        self._create_video_from_frames(frames, os.path.join(sample_dir, f"{base_name}_original.mp4"))
        self._create_video_from_frames(overlays, os.path.join(sample_dir, f"{base_name}_overlay.mp4"))
        mask_frames = [cv2.cvtColor(m, cv2.COLOR_GRAY2BGR) for m in masks]
        self._create_video_from_frames(mask_frames, os.path.join(sample_dir, f"{base_name}_mask.mp4"))

    @staticmethod
    def _create_video_from_frames(frames, output_path, fps: int = 30):
        if not frames:
            return
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        try:
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame.ndim == 3 else frame
                writer.write(frame_bgr)
        finally:
            writer.release()

    # --------------------------------------------------------------- inference
    def process_batch(self, batch: Dict[str, Any], sample_type) -> Dict[str, Any]:
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
                            if batch[key].ndim == 5:
                                sample[key] = batch[key][batch_idx].to(self.device)
                            else:
                                sample[key] = batch[key].to(self.device)
                    output.append(self.model(**sample))
                output = torch.stack(output)
            return {'mask': output, 'input': batch['image']}

    def process_single_sample(self, batch: Dict[str, Any]):
        sample_name = batch['name'][0]
        sample_type = batch['type'][0]
        origin_shape = tuple(batch['origin_shape'].tolist())[0]
        print(f"Processing {sample_type}: {sample_name}")

        base_name = os.path.splitext(sample_name)[0]
        sample_dir = os.path.join(self.output_dir, base_name)
        os.makedirs(sample_dir, exist_ok=True)

        results = self.process_batch(batch, sample_type)
        input_tensor = results['input']
        mask_tensor = results['mask']

        if sample_type == 'image':
            image = self._denormalize_image(input_tensor)
            mask = self._process_mask(mask_tensor)
            image = self._crop_image_to_original(image, origin_shape)
            mask = self._crop_mask_to_original(mask, origin_shape)
            overlay = self._create_overlay(image, mask)
            self._save_image_results(sample_name, sample_dir, image, mask, overlay)
        else:
            if input_tensor.ndim == 4:  # single-frame treated as video
                input_tensor = input_tensor.unsqueeze(1)
                mask_tensor = mask_tensor.unsqueeze(1) if mask_tensor.ndim == 4 else mask_tensor

            frames, masks, overlays = [], [], []
            B, T = input_tensor.shape[0], input_tensor.shape[1]
            for b in range(B):
                for t in range(T):
                    frame = self._denormalize_image(input_tensor[b:b + 1, t])
                    mask = self._process_mask(mask_tensor[b:b + 1, t])
                    frame = self._crop_image_to_original(frame, origin_shape)
                    mask = self._crop_mask_to_original(mask, origin_shape)
                    overlays.append(self._create_overlay(frame, mask))
                    frames.append(frame)
                    masks.append(mask)
            self._save_video_results(sample_name, sample_dir, frames, masks, overlays)

    def run(self, data_path: str, **dataloader_kwargs):
        dataloader = create_inference_dataloader(data_path=data_path, batch_size=1, **dataloader_kwargs)
        print(f"Processing {len(dataloader)} samples...")
        print(f"Output directory: {self.output_dir}")
        for batch in tqdm(dataloader, desc="Processing"):
            self.process_single_sample(batch)
        print("Processing completed!")


def create_model(model_path: str, input_size: int = 1024, grid_size: int = 0):
    """Instantiate RelayFormer, load a checkpoint, and merge LoRA for fast inference."""
    model = RelayFormer(input_size=input_size, grid_size=grid_size)
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.merge_lora()
    return model


def main():
    parser = argparse.ArgumentParser(description='Run RelayFormer inference on images and videos')
    parser.add_argument('--input_dir', type=str, required=True, help='Input file or directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--input_size', type=int, default=1024, help='Model working resolution')
    parser.add_argument('--grid_size', type=int, default=0,
                        help='Tiling; 0 auto-derives as input_size // 512')
    parser.add_argument('--clip_len', type=int, default=4, help='Video clip length')
    parser.add_argument('--overlay_alpha', type=float, default=0.3, help='Overlay transparency')
    parser.add_argument('--mask_threshold', type=float, default=0.5, help='Mask binarization threshold')
    args = parser.parse_args()

    model = create_model(args.model_path, input_size=args.input_size, grid_size=args.grid_size)
    processor = InferenceProcessor(
        model=model, output_dir=args.output_dir, device=args.device,
        overlay_alpha=args.overlay_alpha, mask_threshold=args.mask_threshold)
    processor.run(
        data_path=args.input_dir,
        output_size=(args.input_size, args.input_size),
        clip_len=args.clip_len,
        normalize=True)


if __name__ == "__main__":
    main()
