
import os
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytorch_lightning as pl

# sal_cb = SaliencyCallback(
        #output_dir='outputs/saliency',
        #num_images=4,
        #mean=tuple(config['data']['normalize_mean']),
        #std=tuple(config['data']['normalize_std']),
        #target_layer_path='model.layer4.1.conv2',  # for ResNet18
        #overlay_alpha=0.2,
    #)

# callbacks=[checkpoint_callback, early_stopping, sal_cb],

def _denorm(img: torch.Tensor, mean, std):
    mean = torch.tensor(mean, device=img.device).view(-1, 1, 1)
    std = torch.tensor(std, device=img.device).view(-1, 1, 1)
    x = img * std + mean
    x = x.clamp(0, 1)
    return x

def _to_uint8(img: torch.Tensor):
    return (img.detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)

def _overlay_heatmap(img_rgb_uint8: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.35):
    cmap = plt.get_cmap("jet")
    heat_rgba = cmap(heatmap_01)
    heat_rgb = (heat_rgba[..., :3] * 255).astype(np.uint8)
    if heat_rgb.shape[:2] != img_rgb_uint8.shape[:2]:
        heat_rgb = np.array(Image.fromarray(heat_rgb).resize((img_rgb_uint8.shape[1], img_rgb_uint8.shape[0]), Image.BILINEAR))
    overlay = (alpha * heat_rgb + (1 - alpha) * img_rgb_uint8).astype(np.uint8)
    return overlay

class GradCAMComputer:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_hook = target_layer.register_forward_hook(self._save_activations)
        # register_full_backward_hook is fine on recent torch; fall back if needed
        try:
            self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)
        except AttributeError:
            self.bwd_hook = target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def __call__(self, inputs: torch.Tensor, class_idxs: Optional[List[int]] = None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        if class_idxs is None:
            class_idxs = logits.argmax(dim=1).tolist()
        idxs = torch.tensor(class_idxs, device=inputs.device)
        scores = logits[torch.arange(inputs.size(0), device=inputs.device), idxs].sum()
        scores.backward(retain_graph=True)
        acts = self.activations
        grads = self.gradients
        weights = grads.flatten(2).mean(dim=2)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * acts).sum(dim=1)
        cam = F.relu(cam)
        cam = cam - cam.flatten(1).amin(dim=1, keepdim=True).unsqueeze(-1)
        cam = cam / (cam.flatten(1).amax(dim=1, keepdim=True).unsqueeze(-1) + 1e-8)
        cam = F.interpolate(cam.unsqueeze(1), size=inputs.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return cam

def compute_vanilla_saliency(model: nn.Module, inputs: torch.Tensor, class_idxs: Optional[List[int]] = None):
    x = inputs.clone().detach().requires_grad_(True)
    logits = model(x)
    if class_idxs is None:
        class_idxs = logits.argmax(dim=1).tolist()
    idxs = torch.tensor(class_idxs, device=x.device)
    scores = logits[torch.arange(x.size(0), device=x.device), idxs].sum()
    model.zero_grad(set_to_none=True)
    scores.backward()
    grads = x.grad
    sal = grads.abs().max(dim=1).values
    sal_min = sal.flatten(1).min(dim=1, keepdim=True).values.unsqueeze(-1)
    sal_max = sal.flatten(1).max(dim=1, keepdim=True).values.unsqueeze(-1)
    sal = (sal - sal_min) / (sal_max - sal_min + 1e-8)
    return sal

def compute_smoothgrad_saliency(model, inputs, class_idxs=None, n_samples=25, noise_std=0.1):
    device = inputs.device
    B, C, H, W = inputs.shape
    if class_idxs is None:
        with torch.no_grad():
            preds = model(inputs)
        class_idxs = preds.argmax(dim=1).tolist()
    idxs = torch.tensor(class_idxs, device=device)

    all_sals = []
    for _ in range(n_samples):
        noise = torch.randn_like(inputs) * noise_std
        x_noisy = (inputs + noise).clone().detach().requires_grad_(True)

        logits = model(x_noisy)
        scores = logits[torch.arange(B, device=device), idxs].sum()
        model.zero_grad(set_to_none=True)
        scores.backward()

        grads = x_noisy.grad.abs().max(dim=1).values
        all_sals.append(grads)

    sal = torch.stack(all_sals, dim=0).mean(dim=0)

    # --- robust per-sample normalization ---
    sal_flat = sal.view(B, -1)
    smin = sal_flat.min(dim=1, keepdim=True).values
    smax = sal_flat.max(dim=1, keepdim=True).values
    sal_norm = (sal_flat - smin) / (smax - smin + 1e-8)
    sal = sal_norm.view(B, H, W)
    return sal



class SaliencyCallback(pl.Callback):  # NEW: inherit from pl.Callback
    def __init__(self,
                 output_dir: str = "outputs/saliency",
                 num_images: int = 8,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 target_layer_path: str = "model.layer4.1.conv2",
                 overlay_alpha: float = 0.35):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.num_images = num_images
        self.mean = mean
        self.std = std
        self.target_layer_path = target_layer_path
        self.overlay_alpha = overlay_alpha

    def _resolve_target_layer(self, pl_module):
        m = pl_module.model
        obj = m
        for attr in self.target_layer_path.split('.'):
            if attr.isdigit():
                obj = obj[int(attr)]
            else:
                obj = getattr(obj, attr)
        return obj

    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        device = pl_module.device

        # --- get one validation batch ---
        val_loader = trainer.datamodule.val_dataloader()
        batch = next(iter(val_loader))
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)

        # --- compute predictions WITHOUT grads (fast) ---
        with torch.no_grad():
            logits = pl_module(images)
            preds = logits.argmax(dim=1)

        # --- select N samples we’ll visualize ---
        n = min(self.num_images, images.size(0))
        imgs = images[:n]          # <-- make sure this is defined BEFORE saliency
        lbls = labels[:n]
        prds = preds[:n].tolist()

        # --- prepare output dir ---
        out_dir = self.output_dir / f"epoch_{trainer.current_epoch:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- compute Grad-CAM + vanilla saliency WITH grads enabled ---
        with torch.enable_grad():
            pl_module.zero_grad(set_to_none=True)

            # Grad-CAM
            target_layer = self._resolve_target_layer(pl_module)
            cam_computer = GradCAMComputer(pl_module, target_layer)
            cams = cam_computer(imgs, class_idxs=prds).detach().cpu().numpy()
            cam_computer.remove_hooks()

            # Vanilla saliency
            pl_module.zero_grad(set_to_none=True)
            sal = compute_vanilla_saliency(pl_module, imgs, class_idxs=prds).detach().cpu().numpy()

            # SmoothGrad saliency  <-- NEW
            pl_module.zero_grad(set_to_none=True)
            smooth = compute_smoothgrad_saliency(
                pl_module, imgs, class_idxs=prds,
                n_samples=25,     # tweakable
                noise_std=0.10    # tweakable (std of N(0, σ^2))
            ).detach().cpu().numpy()

        # --- save images & a grid ---
        for i in range(n):
            img_denorm = _denorm(imgs[i], self.mean, self.std)
            base = _to_uint8(img_denorm)

            cam_overlay = _overlay_heatmap(base, cams[i],   alpha=self.overlay_alpha)
            sal_overlay = _overlay_heatmap(base, sal[i],    alpha=self.overlay_alpha)
            smooth_overlay = _overlay_heatmap(base, smooth[i], alpha=self.overlay_alpha)  # NEW

            Image.fromarray(base).save(out_dir / f"sample_{i:02d}_image.png")
            Image.fromarray(cam_overlay).save(out_dir / f"sample_{i:02d}_gradcam.png")
            Image.fromarray(sal_overlay).save(out_dir / f"sample_{i:02d}_saliency.png")
            Image.fromarray(smooth_overlay).save(out_dir / f"sample_{i:02d}_smoothgrad.png")  # NEW


        cols = 4                      # was 3
        rows = n
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3.2, rows*3.2))
        if rows == 1:
            axes = np.expand_dims(axes, 0)
        for i in range(n):
            img_base = _to_uint8(_denorm(imgs[i], self.mean, self.std))
            axes[i,0].imshow(img_base); axes[i,0].set_title(f"GT {int(lbls[i].item())} Pred {int(prds[i])}")
            axes[i,1].imshow(_overlay_heatmap(img_base, cams[i],   alpha=self.overlay_alpha)); axes[i,1].set_title("Grad-CAM")
            axes[i,2].imshow(_overlay_heatmap(img_base, sal[i],    alpha=self.overlay_alpha)); axes[i,2].set_title("Vanilla")
            axes[i,3].imshow(_overlay_heatmap(img_base, smooth[i], alpha=self.overlay_alpha)); axes[i,3].set_title("SmoothGrad")  # NEW
            for j in range(cols):
                axes[i,j].axis("off")
        plt.tight_layout()
        fig.savefig(out_dir / "grid.png", bbox_inches="tight")
        plt.close(fig)


