# nnpu_training.py  (NaN-safe version)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

# -----------------------------------------------------
# 1. Model
# -----------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch, base=32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*8, base*16)
        self.up4 = nn.ConvTranspose2d(base*16, base*8, 2, stride=2)
        self.dec4 = ConvBlock(base*16, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, 1, 1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)

# -----------------------------------------------------
# 2. nnPU Loss + TV Loss (NaN-safe)
# -----------------------------------------------------
def bce_logits(pred_logits, target):
    return F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

def nnpu_loss(logits, label_mask, pi_p, eps=1e-7):
    """Non-negative PU loss (NaN-safe, ignores invalid pixels)."""
    logits = logits.view(-1)
    mask = label_mask.view(-1)

    # Ignore invalid pixels (e.g. mask == -1)
    valid_idx = torch.isfinite(mask) & (mask >= 0)
    if valid_idx.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    logits = logits[valid_idx]
    mask = mask[valid_idx]

    pos_idx = (mask == 1)
    unl_idx = (mask == 0)
    if pos_idx.sum() == 0 or unl_idx.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    loss_pos_pos = bce_logits(logits[pos_idx], torch.ones_like(logits[pos_idx]))
    loss_pos_neg = bce_logits(logits[pos_idx], torch.zeros_like(logits[pos_idx]))
    loss_unl_neg = bce_logits(logits[unl_idx], torch.zeros_like(logits[unl_idx]))

    R_p_pos = loss_pos_pos.mean()
    R_p_neg = loss_pos_neg.mean()
    R_u = loss_unl_neg.mean()

    risk = pi_p * R_p_pos + torch.clamp(R_u - pi_p * R_p_neg, min=0.0)
    return risk

def total_variation_loss(pred_probs):
    dh = torch.abs(pred_probs[:, :, 1:, :] - pred_probs[:, :, :-1, :])
    dw = torch.abs(pred_probs[:, :, :, 1:] - pred_probs[:, :, :, :-1])
    return (dh.mean() + dw.mean())

# -----------------------------------------------------
# 3. Dataset + Patch Sampler (NaN-safe)
# -----------------------------------------------------
class PositiveUnlabeledDataset(Dataset):
    def __init__(self, images, labels, add_valid_mask=True):
        # Fill NaNs and optionally add validity mask as extra channel
        X = np.copy(images)
        Y = np.copy(labels)

        valid_mask = (~np.isnan(X)).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)

        if add_valid_mask:
            X = np.concatenate([X, valid_mask[:, :1, :, :]], axis=1)

        # Replace NaNs in labels with -1 (ignored)
        Y = np.nan_to_num(Y, nan=-1.0)

        self.images = torch.from_numpy(X).float()
        self.labels = torch.from_numpy(Y).float()

    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class PatchSampler:
    """Samples patches, avoiding NaN-only regions."""
    def __init__(self, patch_size=256, pos_fraction=0.5):
        self.patch_size = patch_size
        self.pos_fraction = pos_fraction

    def collate(self, batch):
        imgs, lbls = zip(*batch)
        imgs = torch.stack(imgs)
        lbls = torch.stack(lbls)
        _, _, H, W = imgs.shape
        ps = self.patch_size

        patches_img, patches_lbl = [], []
        for i in range(imgs.shape[0]):
            valid = torch.isfinite(imgs[i]).any(dim=0)
            if valid.sum() == 0:
                continue  # skip fully invalid image

            # sample patch center
            if torch.rand(1).item() < self.pos_fraction and (lbls[i] == 1).sum() > 0:
                pos = (lbls[i][0] == 1).nonzero()
                yx = pos[torch.randint(0, len(pos), (1,))][0]
                y, x = int(yx[0]), int(yx[1])
            else:
                y, x = torch.randint(ps//2, H-ps//2, (1,)).item(), torch.randint(ps//2, W-ps//2, (1,)).item()

            y0, x0 = max(0, y-ps//2), max(0, x-ps//2)
            y1, x1 = min(H, y0+ps), min(W, x0+ps)
            patches_img.append(imgs[i,:,y0:y1,x0:x1])
            patches_lbl.append(lbls[i,:,y0:y1,x0:x1])

        if len(patches_img) == 0:
            # fallback to full image if all invalid
            return imgs, lbls
        return torch.stack(patches_img), torch.stack(patches_lbl)

# -----------------------------------------------------
# 4. Estimate πₚ (Elkan & Noto, NaN-safe)
# -----------------------------------------------------
def estimate_pi_p(dataloader, model, device, warmup_epochs=3):
    """Estimate positive prior πₚ using Elkan & Noto (2008)."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    # Quick warmup classifier: positives vs unlabeled
    for epoch in range(warmup_epochs):
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            valid_idx = (labels >= 0)
            if valid_idx.sum() == 0:
                continue
            loss = loss_fn(logits[valid_idx], labels[valid_idx])
            loss.backward()
            optimizer.step()

    # Estimate πₚ = E[f(x_p)] over labeled positives
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            probs = torch.sigmoid(model(imgs))
            valid_idx = (labels == 1)
            if valid_idx.sum() == 0:
                continue
            preds.append(probs[valid_idx].cpu())
    if len(preds) == 0:
        return 0.01  # fallback small prior
    preds = torch.cat(preds)
    return preds.mean().item()
