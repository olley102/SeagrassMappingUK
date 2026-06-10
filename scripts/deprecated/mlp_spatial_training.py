# mlp_spatial_training.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import precision_recall_curve, auc

# -----------------------------------------------------
# 1. Model
# -----------------------------------------------------
class CenterNeighborhoodMLP(nn.Module):
    def __init__(self, in_channels, patch_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.center_idx = patch_size // 2
        flat_size = in_channels * patch_size * patch_size
        
        # Duplicate center pixel 8 times (strong inductive bias)
        self.net = nn.Sequential(
            nn.Linear(flat_size + in_channels * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        def forward(self, x):
        B, C, H, W = x.shape
        flat = x.flatten(2)  # (B, C, H*W)

        # Extract and duplicate center pixel
        center_pixel = x[
            :, :, self.center_idx, self.center_idx
        ]  # (B, C)
        center_dup = center_pixel.repeat(1, 8)  # (B, C*8)

        # Concatenate
        enhanced = torch.cat([flat, center_dup], dim=1)
        return self.net(enhanced)


# -----------------------------------------------------
# 4. Dataset
# -----------------------------------------------------
class PositiveUnlabeledDataset(Dataset):
    def __init__(self, images, labels):
        """
        Accepts either:
          - np.ndarray: in-memory array
          - np.memmap: memory-mapped array (lazy loading)
        images: shape (N, C, H, W)
        labels: shape (N, 1, H, W)
        """
        self.images_source = images
        self.labels_source = labels

        # Determine type and length
        if isinstance(images, np.memmap):
            self.is_memmap = True
            self.num_samples = len(images)
        elif isinstance(images, np.ndarray):
            self.is_memmap = False
            self.num_samples = images.shape[0]
        else:
            raise TypeError("images must be np.ndarray or np.memmap")

        if isinstance(labels, np.memmap):
            if len(labels) != self.num_samples:
                raise ValueError("images and labels must have same number of samples")
        elif isinstance(labels, np.ndarray):
            if labels.shape[0] != self.num_samples:
                raise ValueError("images and labels must have same number of samples")
        else:
            raise TypeError("labels must be np.ndarray or np.memmap")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load data based on type (memmap: lazy, ndarray: direct slice)
        if self.is_memmap:
            img = self.images_source[idx].astype(np.float32)
            lbl = self.labels_source[idx].astype(np.float32)
        else:
            img = self.images_source[idx].copy()
            lbl = self.labels_source[idx].copy()

        # Ensure contiguous arrays for safety
        img = np.ascontiguousarray(img)
        lbl = np.ascontiguousarray(lbl)

        # Compute valid mask from image
        valid_mask = (~np.isnan(img)).all(axis=0, keepdims=True)
        
        # Replace NaNs with 0 in image
        img = np.nan_to_num(img, nan=0.0)

        # Apply invalid mask to label: invalid to -1
        lbl = np.where(valid_mask, lbl, -1.0)
        lbl = np.nan_to_num(lbl, nan=-1.0)

        return torch.from_numpy(img).float(), torch.from_numpy(lbl).float()


class NeighborhoodSampler:
    def __init__(self, patch_size=64, pos_fraction=0.7, invalid_value=-1.0):
        self.patch_size = patch_size
        self.pos_fraction = pos_fraction
        self.invalid_value = invalid_value

    def _extract_patch_with_padding(self, img, cy, cx):
        """
        Extract a patch with zero-padding to ensure output is exactly patch_size x patch_size.
        """
        C, H, W = img.shape
        ps = self.patch_size

        y0 = cy - ps // 2
        y1 = cy + ps // 2 + (ps % 2)
        x0 = cx - ps // 2
        x1 = cx + ps // 2 + (ps % 2)

        # Compute valid region inside image
        iy0 = max(0, y0)
        ix0 = max(0, x0)
        iy1 = min(H, y1)
        ix1 = min(W, x1)

        # Allocate padded tensor
        img_patch = torch.zeros((C, ps, ps), dtype=img.dtype, device=img.device)

        # Compute destination index in patch
        py0 = iy0 - y0
        px0 = ix0 - x0
        py1 = py0 + (iy1 - iy0)
        px1 = px0 + (ix1 - ix0)

        img_patch[:, py0:py1, px0:px1] = img[:, iy0:iy1, ix0:ix1]

        return img_patch

    def collate(self, batch):
        """
        batch: list of (img_tensor, peak_tensor) from dataset
        img:  (C, H, W)
        lbl: (1, H, W) in {1, 0, invalid_value}
        """
        imgs, lbls = zip(*batch)
        imgs = torch.stack(imgs)  # (B, C, H, W)
        lbls = torch.stack(lbls)  # (B, 1, H, W)
        
        B, C, H, W = imgs.shape
        ps = self.patch_size
        patch_imgs, patch_lbls = [], []

        for i in range(B):
            img = imgs[i]
            lbl = lbls[i]
            
            valid_mask = (lbl != self.invalid_value)
            has_pos = (lbl > 0.5) & valid_mask

            # Decide: sample from positive or anywhere?
            if torch.rand(1) < self.pos_fraction:
                # Sample from positives
                candidates = has_pos.nonzero()
                if len(candidates) > 0:
                    idx = torch.randint(0, len(candidates), (1,))
                    cy, cx = candidates[idx][0].tolist()
                else:
                    # Fallback: random
                    cy = torch.randint(ps // 2, H - ps // 2, (1,)).item()
                    cx = torch.randint(ps // 2, W - ps // 2, (1,)).item()
            else:
                # Random background patch
                cy = torch.randint(ps // 2, H - ps // 2, (1,)).item()
                cx = torch.randint(ps // 2, W - ps // 2, (1,)).item()

            img_p = self._extract_patch_with_padding(img, cy, cx)
            patch_imgs.append(img_p)
            patch_lbls.append(lbl[:, cy, cx])  # (1,)

        return torch.stack(patch_imgs), torch.stack(patch_lbls)


# -----------------------------------------------------
# 5. Training loop
# -----------------------------------------------------
def train_center_nbhd_mlp(
    model,
    train_dataset,
    device,
    num_epochs=60,
    batch_size=16,
    patch_size=64,
    pos_fraction=0.7,
    invalid_value=-1.0
    lr=1e-3
):

    model = model.to(device)

    # Patch sampler and dataloader
    sampler = NeighborhoodSampler(
        patch_size=patch_size,
        pos_fraction=pos_fraction,
        invalid_value=invalid_value
    )
    train_loader = DataLoader(
        train_dataset,
        collate_fn=sampler.collate,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device).flatten()  # (B,)

            optimizer.zero_grad()

            logits = model(imgs).squeeze()  # (B,)
            
            valid = (lbls != invalid_value)
            
            if valid.any():
                loss = F.binary_cross_entropy_with_logits(
                    logits[valid], lbls[valid]
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_batches += 1
        
        loss_batch = total_loss/total_batches
        print(f"Epoch {epoch+1:02d} | BCE={loss_batch:.6f}")
        history.append(loss_batch)

    return model, history


# -----------------------------------------------------
# 6. Validation & Visualization utilities
# -----------------------------------------------------
@torch.no_grad()
def infer_full_image_mlp(model, img_tensor, device,
                        patch_size=64, stride=32):
    """
    Sliding window inference with overlap blending.
    Returns probability map of shape (H, W)
    """
    model.eval()
    img_tensor = img_tensor.to(device)          # (C, H, W)
    C, H, W = img_tensor.shape
    
    prob_map = torch.zeros((H, W), device=device)
    weight_map = torch.zeros((H, W), device=device)

    half = patch_size // 2

    # Pre-compute Gaussian weighting window (smoother blending)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing='ij'
    )
    center_y, center_x = patch_size // 2, patch_size // 2
    gaussian_weight = torch.exp(
        -4 * ((y_grid - center_y)**2 + (x_grid - center_x)**2) / (patch_size**2)
    )  # falls to ~0.01 at edges

    for cy in range(half, H - half, stride):
        for cx in range(half, W - half, stride):
            y0, y1 = cy - half, cy + half
            x0, x1 = cx + half

            # Handle boundaries
            pad_top = max(0, -y0)
            pad_bottom = max(0, y1 - H)
            pad_left = max(0, -x0)
            pad_right = max(0, x1 - W)

            patch = img_tensor[:, 
                               max(0, y0):min(H, y1),
                               max(0, x0):min(W, x1)]

            if any((pad_top, pad_bottom, pad_left, pad_right)):
                patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom))

            # Forward pass
            logit = model(patch.flatten().unsqueeze(0))
            prob = torch.sigmoid(logit).squeeze(0)

            # Write with Gaussian weighting
            sy, sx = max(0, y0), max(0, x0)
            ey, ex = min(H, y1), min(W, x1)
            py0, px0 = pad_top, pad_left

            prob_map[sy:ey, sx:ex] += prob * gaussian_weight[py0:py0 + ey-sy, px0:px0 + ex-sx]
            weight_map[sy:ey, sx:ex] += gaussian_weight[py0:py0 + ey-sy, px0:px0 + ex-sx]

    prob_map = prob_map / (weight_map + 1e-8)
    return prob_map.cpu().numpy()


@torch.no_grad()
def evaluate_dataset_full_image(model, dataset, device,
                               patch_size=64, stride=32,
                               invalid_value=-1.0):
    """
    Full-image evaluation over entire dataset.
    Returns aggregated PU metrics + per-image dict.
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_valid = []

    results = []

    for idx in range(len(dataset)):
        img, lbl = dataset[idx]                     # (C,H,W), (1,H,W)
        img_tensor = img.to(device)
        lbl_np = lbl.squeeze(0).numpy()             # (H,W)

        pred_prob = infer_full_image_mlp(model, img_tensor, device,
                                         patch_size=patch_size, stride=stride)

        valid_mask = (lbl_np != invalid_value)
        true_pos = (lbl_np > 0.5)

        # Collect only valid pixels
        probs_flat = pred_prob[valid_mask]
        labels_flat = true_pos[valid_mask].astype(float)

        all_probs.extend(probs_flat)
        all_labels.extend(labels_flat)

        # Per-image metrics
        precision, recall, thr = precision_recall_curve(labels_flat, probs_flat)
        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        results.append({
            'image_idx': idx,
            'num_positive_pixels': int(true_pos.sum()),
            'num_valid_pixels': int(valid_mask.sum()),
            'PU_F1': float(f1.max()),
            'PR_AUC': float(auc(recall, precision)),
            'pred_map': pred_prob,
            'true_map': lbl_np
        })

    # Global metrics
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)

    global_metrics = {
        'Global_PU_F1': float(f1_scores.max()),
        'Global_PR_AUC': float(auc(recall, precision)),
        'Global_AP': float(average_precision_score(all_labels, all_probs)),
        'per_image_results': results
    }

    return global_metrics


def visualize_prediction(img_np, true_lbl_np, pred_prob_np,
                         invalid_value=-1.0, threshold=0.5,
                         cmap_img='gray', cmap_label='jet',
                         save_path=None):
    """
    Creates a 1x4 plot:
      1. Input image (channel 0 shown if multi-channel)
      2. Ground-truth labels (positive=1, unlabeled=0, invalid=-1)
      3. Predicted probability
      4. Binary prediction @ threshold
    """
    fig, axs = plt.subplots(1, 4, figsize=(24, 6))

    # 1. Image
    im0 = img_np[0] if img_np.ndim == 3 else img_np
    axs[0].imshow(im0, cmap=cmap_img)
    axs[0].set_title("Input Image (ch0)")
    axs[0].axis('off')

    # 2. True labels
    true_vis = true_lbl_np.copy()
    true_vis[true_lbl_np == invalid_value] = np.nan
    im1 = axs[1].imshow(true_vis.squeeze(), cmap=cmap_label, vmin=0, vmax=1)
    axs[1].set_title("Ground Truth (positive=1, unlabeled=0)")
    axs[1].axis('off')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    # 3. Predicted probability
    im2 = axs[2].imshow(pred_prob_np, cmap='jet', vmin=0, vmax=1)
    axs[2].set_title("Predicted Probability")
    axs[2].axis('off')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # 4. Binary prediction
    binary = (pred_prob_np > threshold).astype(float)
    invalid_mask = (true_lbl_np.squeeze() == invalid_value)
    binary[invalid_mask] = np.nan
    im3 = axs[3].imshow(binary, cmap=cmap_label, vmin=0, vmax=1)
    axs[3].set_title(f"Prediction @ thr={threshold}")
    axs[3].axis('off')
    plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# -----------------------------------------------------
# Optional: simple training loop with validation
# -----------------------------------------------------
def train_with_validation(
    model, train_dataset, val_dataset, device,
    num_epochs=80, batch_size=16, patch_size=64,
    pos_fraction=0.7, lr=1e-3, save_dir="checkpoints"
):
    Path(save_dir).mkdir(exist_ok=True)
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model, _ = train_mlp(
            model=model,
            train_dataset=train_dataset,
            device=device,
            num_epochs=1,                 # we call it epoch-by-epoch
            batch_size=batch_size,
            patch_size=patch_size,
            pos_fraction=pos_fraction,
            lr=lr
        )

        val_metrics = validate_mlp_pu(
            model, val_dataset, device,
            batch_size=batch_size,
            patch_size=patch_size,
            stride=patch_size//2
        )

        print(f"Epoch {epoch+1:02d} | "
              f"Val BCE: {val_metrics['Val_BCE']:.4f} | "
              f"PU-F1: {val_metrics['PU_F1']:.4f} | "
              f"PU-PR-AUC: {val_metrics['PU_PR_AUC']:.4f}")

        # Save best model according to PU-F1
        if val_metrics['PU_F1'] > best_f1:
            best_f1 = val_metrics['PU_F1']
            torch.save(model.state_dict(),
                       os.path.join(save_dir, "best_mlp_pu.pth"))
            print(f"  → New best PU-F1! Model saved.")

    print(f"Training finished. Best PU-F1 = {best_f1:.4f}")
    return model
