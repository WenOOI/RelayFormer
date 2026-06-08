import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np

from IMDLBenCo.registry import MODELS

from .GLoRA_vit import DecAttention, replace_vit_modules, build_4d_position_ids


class GatedMaskDecoder(nn.Module):
    """
    A decoder module that generates masks using cross-attention, self-attention, and feed-forward networks.
    """
    def __init__(self, in_channels: int, num_queries: int, hidden_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.num_layers = num_layers

        # Project input features to hidden dimension
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        # MLP to compute gating weights
        self.gate_mlp = nn.Linear(hidden_dim, 1)
        # Learnable mask embeddings
        self.mask_embed = nn.Embedding(num_queries, hidden_dim)

        # Stack of decoder layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads,
                                                    dropout=0.1, batch_first=True),
                'norm_cross': nn.LayerNorm(hidden_dim),
                'self_attn': DecAttention(dim=hidden_dim, num_heads=num_heads, proj_drop=0.1),
                'norm_self': nn.LayerNorm(hidden_dim),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim)
                ),
                'norm_ffn': nn.LayerNorm(hidden_dim)
            })
            self.layers.append(layer)

        # Final convolution for mask output
        self.final_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, num_patches=4) -> torch.Tensor:
        B, C, H, W = x.shape
        # Project input features
        x_proj = self.input_proj(x)  # (B, hidden_dim, H, W)
        # Expand mask embeddings to batch size
        mask_embed = self.mask_embed.weight.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Flatten input for attention
        x_flat = x_proj.flatten(2).permute(0, 2, 1)  # (B, H*W, hidden_dim)
        kv = x_flat  # Key and value for cross-attention
        q = mask_embed  # Query for cross-attention

        # Process through decoder layers
        for layer in self.layers:
            # Cross-attention
            attn_cross, _ = layer['cross_attn'](query=q, key=kv, value=kv)
            q = layer['norm_cross'](q + attn_cross)
            # Self-attention
            attn_self = layer['self_attn'](x=(q, q, q))
            q = layer['norm_self'](q + attn_self)
            # Feed-forward network
            ffn_out = layer['ffn'](q)
            q = layer['norm_ffn'](q + ffn_out)

        # Compute mask logits
        mask_queries = q  # (B, num_queries, hidden_dim)
        mask_logits = torch.einsum('bnd,bdhw->bnhw', mask_queries, x_proj)  # (B, num_queries, H, W)
        gates = self.gate_mlp(mask_queries).sigmoid()  # (B, num_queries, 1)
        final_mask = (mask_logits * gates.unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B, 1, H, W)

        return final_mask


@MODELS.register_module()
class RelayFormer(nn.Module):
    """
    RelayFormer: unified local-global attention framework for image & video manipulation localization.

    This is the optimized model (batched global attention + F.fold reconstruction + LoRA merging)
    wired into the IMDLBenCo training/testing pipeline. The same `forward` serves three purposes:
      - training: when `mask` is provided, returns an output dict with `backward_loss`.
      - testing: same path (IMDLBenCo passes `mask`); the returned `pred_mask` is used by evaluators.
      - pure inference: when `mask` is None, returns the predicted mask tensor directly.

    Image-only / video-only / mixed training is handled entirely by the dataset; the model just
    sees a flattened batch of frames described by `clip_len` and `origin_shape`.
    """
    def __init__(self,
                 input_size: int = 1024,
                 vit_model_name: str = 'vit_base_patch16_224',
                 vit_pretrain_path: str = '',
                 tokens_per_patch: int = 2,
                 edge_lambda: int = 20,
                 grid_size: int = 0,
                 patch_size: int = 528,
                 overlap: int = 16,
                 feature_patch_size: int = 33,
                 feature_overlap: int = 1):
        """
        Args (auto-exposed as CLI flags by the IMDLBenCo argparser via type annotations):
            input_size: model working resolution (must equal dataset image_size).
            vit_model_name: timm ViT backbone name.
            vit_pretrain_path: path to a ViT checkpoint (.bin/.pth). Empty -> timm auto-download.
            tokens_per_patch: number of relay tokens per patch.
            edge_lambda: weight for the edge-aware BCE loss term.
            grid_size: max NxN tiling per image. 0 (default) -> derive as input_size // 512.
                Set explicitly (e.g. 2) to reproduce a fixed tiling at a non-1024 resolution.
            patch_size / overlap: image-space patch tiling.
            feature_patch_size / feature_overlap: feature-space patch tiling.

        Resolution presets (input_size : patch_size / feature_patch_size : grid_size):
            1024 : 528 / 33 : 2   (default; grid auto-derived)
            512  : 272 / 17 : 2   (pass --grid_size 2 explicitly)
        """
        super().__init__()

        self.input_size = input_size
        self.edge_lambda = edge_lambda

        # Patch parameters
        self.patch_size = patch_size
        self.overlap = overlap
        self.feature_patch_size = feature_patch_size
        self.feature_overlap = feature_overlap

        # Compute strides
        self.stride = self.patch_size - self.overlap * 2
        self.feature_stride = self.feature_patch_size - self.feature_overlap * 2

        # Create Vision Transformer. When no checkpoint path is given we let timm pull its
        # pretrained ImageNet weights so the (replaced) blocks still start from a good init.
        use_timm_pretrained = (vit_pretrain_path is None or vit_pretrain_path == '')
        vit = timm.create_model(vit_model_name, pretrained=use_timm_pretrained, num_classes=0,
                                reg_tokens=1, drop_path_rate=0.1, dynamic_img_size=True,
                                no_embed_class=False)
        pretrained_state = vit.state_dict() if use_timm_pretrained else None

        # Replace stock blocks with LoRA + 4D-RoPE blocks (parameter names for the shared
        # qkv/proj/fc layers are preserved, so pretrained weights can be loaded afterwards).
        self.vit = replace_vit_modules(vit)
        self.feature_dim = self.vit.num_features

        if use_timm_pretrained:
            # Re-load timm pretrained weights into the replaced blocks (LoRA params stay random).
            missing, unexpected = self.vit.load_state_dict(pretrained_state, strict=False)
            print(f"[RelayFormer] Loaded timm pretrained ViT. missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            self._load_vit_checkpoint(vit_pretrain_path)

        self.tpp = tokens_per_patch
        # 0 -> auto-derive (1024 yields 2). Override to pin a fixed tiling at other resolutions.
        self.grid_size = grid_size if grid_size > 0 else input_size // 512

        # Initialize mask decoder
        self.mask_decoder = GatedMaskDecoder(
            in_channels=self.feature_dim,
            num_queries=8,
            hidden_dim=256,
            num_layers=3,
        )

        # Binary cross-entropy loss for training
        self.BCE_loss = nn.BCEWithLogitsLoss()

    def _load_vit_checkpoint(self, path: str):
        """Load a ViT checkpoint, adapting pos_embed for the extra register token if needed."""
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        state_dict = state_dict.get('model', state_dict)
        # Strip a possible "vit." prefix.
        state_dict = {k[4:] if k.startswith('vit.') else k: v for k, v in state_dict.items()}

        if 'pos_embed' in state_dict:
            ckpt_len = state_dict['pos_embed'].shape[1]
            model_len = self.vit.pos_embed.shape[1]
            # Checkpoint lacks the register-token slot -> insert one after the cls token.
            if ckpt_len == model_len - 1:
                orig_pos = state_dict['pos_embed']
                embed_dim = orig_pos.shape[2]
                new_pe = torch.randn(1, 1, embed_dim) * 0.02
                state_dict['pos_embed'] = torch.cat([orig_pos[:, :1, :], new_pe, orig_pos[:, 1:, :]], dim=1)

        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        print(f"[RelayFormer] Loaded ViT checkpoint from {path}. "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

    def merge_lora(self):
        """Merge all LoRA weights for faster inference. Call before eval."""
        for block in self.vit.blocks:
            block.merge_lora()

    def divide_patches(self, x):
        """Divide input image into patches using unfold. Returns (B, N, C, ph, pw)."""
        patches = x.unfold(2, self.patch_size, self.stride).unfold(3, self.patch_size, self.stride)
        B, C, n_h, n_w, ph, pw = patches.shape
        return patches.permute(0, 2, 3, 1, 4, 5).reshape(B, n_h * n_w, C, ph, pw)

    def _batched_global_attention(self, relay_tokens_list, sample_infos, block):
        """
        Process relay tokens from all samples in a single batched forward pass.

        Returns a list of processed tensors, each [num_patches_i, tpp, C].
        """
        num_samples = len(relay_tokens_list)
        tpp = self.tpp

        if num_samples == 1:
            info = sample_infos[0]
            processed_flat = block(
                relay_tokens_list[0], use_lora=1,
                valid=info['valid'],
                grid_size=self.grid_size,
                clip_len=info['clip_len']
            )
            return [processed_flat.view(info['num_patches'], tpp, -1)]

        device = relay_tokens_list[0].device
        dtype = relay_tokens_list[0].dtype
        C = relay_tokens_list[0].shape[2]

        lengths = [r.shape[1] for r in relay_tokens_list]
        max_len = max(lengths)

        padded = torch.zeros(num_samples, max_len, C, device=device, dtype=dtype)
        for i, (r, l) in enumerate(zip(relay_tokens_list, lengths)):
            padded[i, :l] = r[0]

        attn_mask = torch.zeros(num_samples, 1, max_len, max_len, device=device, dtype=dtype)
        for i, l in enumerate(lengths):
            if l < max_len:
                attn_mask[i, :, :l, l:] = float('-inf')
                attn_mask[i, :, l:, :] = 0

        position_ids = torch.zeros(4, num_samples, max_len, device=device, dtype=torch.long)
        for i, info in enumerate(sample_infos):
            pid = build_4d_position_ids(
                info['clip_len'], info['valid'], self.grid_size, tpp, device=device)
            position_ids[:, i, :lengths[i]] = pid[:, 0, :]

        processed = block(
            padded, use_lora=1,
            attn_mask=attn_mask, position_ids_4d=position_ids
        )

        results = []
        for i, info in enumerate(sample_infos):
            results.append(processed[i, :lengths[i]].view(info['num_patches'], tpp, C))
        return results

    def forward_features(self, patches_list, valid_patches_list, grid_hw_list, clip_len_list):
        """Process patches through the ViT and reconstruct features."""
        all_patches = []
        patch_counts = []
        for patches in patches_list:
            all_patches.append(patches)
            patch_counts.append(patches.shape[0])

        x = torch.cat(all_patches, dim=0)

        # Vision Transformer forward pass
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        x = self.vit.patch_drop(x)
        x = self.vit.norm_pre(x)

        B, N, C = x.shape
        tpp = self.tpp
        features = []

        for i, blk in enumerate(self.vit.blocks):
            x = blk(x, use_lora=0, grid_size=self.grid_size)

            # Phase 1: Collect relay tokens and reconstruct local tokens per sample
            relay_tokens_list = []
            loc_tokens_list = []
            sample_infos = []

            start_idx = 0
            batch_idx = 0
            video_batch = 0
            while start_idx < x.shape[0]:
                clip_len = clip_len_list[batch_idx].item()
                patch_count = patch_counts[video_batch]
                num_patches = patch_count * clip_len
                end_idx = start_idx + num_patches

                x_batch = x[start_idx:end_idx]
                loc_x_i = x_batch[:, tpp:]

                loc_reconstructed = self._reconstruct_multi_frame(
                    loc_x_i, valid_patches_list[video_batch],
                    grid_hw_list[video_batch], clip_len, patch_count)
                loc_tokens_list.append(loc_reconstructed)

                relay = x_batch[:, :tpp].contiguous().view(1, num_patches * tpp, C)
                relay_tokens_list.append(relay)

                sample_infos.append({
                    'num_patches': num_patches,
                    'valid': valid_patches_list[video_batch],
                    'clip_len': clip_len,
                })

                start_idx = end_idx
                batch_idx += 1
                video_batch += clip_len

            # Phase 2: Batched global attention on relay tokens
            if i + 1 < len(self.vit.blocks):
                processed_relays = self._batched_global_attention(
                    relay_tokens_list, sample_infos, self.vit.blocks[i+1])
            else:
                processed_relays = [r.squeeze(0).view(info['num_patches'], tpp, C)
                                    for r, info in zip(relay_tokens_list, sample_infos)]

            # Phase 3: Reassemble
            x_processed_list = []
            for idx in range(len(sample_infos)):
                x_batch_reshaped = torch.cat(
                    [processed_relays[idx], loc_tokens_list[idx]], dim=1)
                x_processed_list.append(x_batch_reshaped)

            x = torch.cat(x_processed_list, dim=0)

        x = self.vit.norm(x)
        for _ in range(4):
            features.append(x[:, self.tpp:])
        return x, features

    def get_valid_patches_info(self, original_shapes, input_size=1036, grid_size=2, clip_len=None):
        """Compute valid patch indices and grid information."""
        shapes = np.array(original_shapes.cpu(), dtype=float)
        scales = shapes / input_size
        needed = np.ceil(scales * grid_size).astype(int)
        needed = np.clip(needed, 1, grid_size)

        cache = {}
        for h in range(1, grid_size + 1):
            for w in range(1, grid_size + 1):
                cache[(h, w)] = [i * grid_size + j for i in range(h) for j in range(w)]

        valid_patches_list = []
        grid_hw_list = []
        for (h, w), clip_l in zip(needed, clip_len):
            valid_patches_list.extend([cache[(h.item(), w.item())]] * clip_l)
            grid_hw_list.extend([(h.item(), w.item())] * clip_l)

        max_valid_patches = max(len(p) for p in valid_patches_list)
        return valid_patches_list, max_valid_patches, grid_hw_list

    @staticmethod
    def select_valid_patches(valid, patches):
        """Select valid patches based on indices."""
        patches_list = []
        for i in range(len(valid)):
            valid_indices = valid[i]
            if len(valid_indices) > 0:
                selected_patches = patches[i, valid_indices]
            else:
                raise ValueError("Empty input image.")
            patches_list.append(selected_patches)
        return patches_list

    def _get_recon_dims(self, grid_hw):
        """Compute reconstructed feature map dimensions for a given grid layout."""
        grid_h, grid_w = grid_hw
        H = self.feature_patch_size if grid_h == 1 else grid_h * self.feature_stride + self.feature_overlap * 2
        W = self.feature_patch_size if grid_w == 1 else grid_w * self.feature_stride + self.feature_overlap * 2
        return H, W

    def reconstruct_from_overlap_patches(self, patches, valid_indices, grid_hw, for_decode=False):
        """Reconstruct feature map from overlapping patches using F.fold."""
        num_patches, N, C = patches.shape
        fp = self.feature_patch_size
        H, W = self._get_recon_dims(grid_hw)

        fold_input = (patches.view(num_patches, fp, fp, C)
                      .permute(3, 1, 2, 0)
                      .reshape(1, C * fp * fp, num_patches))

        recon = F.fold(fold_input, output_size=(H, W), kernel_size=fp, stride=self.feature_stride)

        ones = torch.ones(1, fp * fp, num_patches, device=patches.device, dtype=patches.dtype)
        count = F.fold(ones, output_size=(H, W), kernel_size=fp, stride=self.feature_stride)
        recon = recon / torch.clamp(count, min=1.0)

        if for_decode:
            return recon

        patches_reconstructed = self.divide_feature_patches(recon)
        return patches_reconstructed.view(num_patches, C, N).permute(0, 2, 1)

    def _reconstruct_multi_frame(self, patches, valid_indices, grid_hw, clip_len, patch_count):
        """Batched reconstruction for multiple frames sharing the same grid layout."""
        N, C = patches.shape[1], patches.shape[2]
        fp = self.feature_patch_size
        H, W = self._get_recon_dims(grid_hw)

        fold_input = (patches.view(clip_len, patch_count, fp, fp, C)
                      .permute(0, 4, 2, 3, 1)
                      .reshape(clip_len, C * fp * fp, patch_count))

        recon = F.fold(fold_input, output_size=(H, W), kernel_size=fp, stride=self.feature_stride)

        ones = torch.ones(1, fp * fp, patch_count, device=patches.device, dtype=patches.dtype)
        count = F.fold(ones, output_size=(H, W), kernel_size=fp, stride=self.feature_stride)
        recon = recon / torch.clamp(count, min=1.0)

        patches_out = self.divide_feature_patches(recon)
        return patches_out.reshape(clip_len * patch_count, C, N).permute(0, 2, 1)

    def divide_feature_patches(self, x):
        """Divide feature map into feature patches using unfold."""
        fp = self.feature_patch_size
        patches = x.unfold(2, fp, self.feature_stride).unfold(3, fp, self.feature_stride)
        B, C, n_h, n_w, ph, pw = patches.shape
        return patches.permute(0, 2, 3, 1, 4, 5).reshape(B, n_h * n_w, C, ph, pw)

    def assemble_and_decode(self, x, images, valid_patches_list, grid_hw_list, grid_size=2):
        """Assemble patches and decode to generate masks. Returns logits (B, 1, input_size, input_size)."""
        device = x.device
        B = len(valid_patches_list)
        full_masks = torch.zeros(B, 1, self.input_size, self.input_size, device=device) - 10.

        stx = 0
        for i in range(B):
            edy = stx + len(valid_patches_list[i])
            ix = self.reconstruct_from_overlap_patches(x[stx:edy], valid_patches_list[i],
                                                       grid_hw_list[i], for_decode=True)
            h, w = grid_hw_list[i]
            pred_mask = self.mask_decoder(ix)
            h = h * self.input_size // self.grid_size
            w = w * self.input_size // self.grid_size
            pred_mask = F.interpolate(pred_mask, size=(int(h), int(w)), mode='bilinear', align_corners=False)
            full_masks[i:i+1, :, :h, :w] = pred_mask
            stx = edy

        return full_masks

    def forward(self, image, clip, origin_shape, clip_len,
                mask=None, edge_mask=None, label=None, *args, **kwargs):
        """
        Args:
            image:        (B, C, H, W) middle/representative frame per sample (assembled batch).
            clip:         (B, C, H, W) frames flattened across the batch.
            origin_shape: (B, 2) original (H, W) per sample.
            clip_len:     (num_samples,) frames-per-sample; 1 for images, T for video clips.
            mask:         (B, 1, H, W) ground-truth mask. If None -> pure inference mode.
            edge_mask:    (B, 1, H, W) optional edge weight map for the edge-aware loss.

        Returns:
            - dict (IMDLBenCo output interface) when `mask` is provided.
            - predicted mask tensor (B, 1, H, W) when `mask` is None.
        """
        grid_size = self.grid_size
        patches = self.divide_patches(clip)
        valid_patches_list, max_valid_patches, grid_hw_list = self.get_valid_patches_info(
            origin_shape, self.input_size, grid_size, clip_len=clip_len)
        patches = self.select_valid_patches(valid_patches_list, patches)
        features, feature_list = self.forward_features(
            patches, valid_patches_list, grid_hw_list, clip_len_list=clip_len)
        pred_logits = self.assemble_and_decode(
            features[:, self.tpp:], image, valid_patches_list, grid_hw_list, grid_size)

        mask_pred = torch.sigmoid(pred_logits)

        # Pure inference (e.g. infer.py): no ground truth available.
        if mask is None:
            return mask_pred

        # Training / testing: compute losses and return the IMDLBenCo output dict.
        loss_bce = self.BCE_loss(pred_logits, mask)
        if edge_mask is not None:
            edge_loss = F.binary_cross_entropy_with_logits(
                input=pred_logits, target=mask, weight=edge_mask) * self.edge_lambda
        else:
            edge_loss = torch.zeros((), device=pred_logits.device)
        combined_loss = loss_bce + edge_loss

        output_dict = {
            "backward_loss": combined_loss,
            "pred_mask": mask_pred,
            "pred_label": None,
            "visual_loss": {
                "bce_loss": loss_bce,
                "edge_loss": edge_loss,
                "combined_loss": combined_loss,
            },
            "visual_image": {
                "pred_mask": mask_pred,
            },
        }
        if edge_mask is not None:
            output_dict["visual_image"]["edge_mask"] = edge_mask

        return output_dict


if __name__ == "__main__":
    print(MODELS)
