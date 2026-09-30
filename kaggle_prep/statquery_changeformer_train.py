# ============================================================
#  StatQuery AI -- ChangeFormer Fine-Tuning on LEVIR-CD
#  Kaggle Notebook  |  GPU: P100 / T4  (16 GB VRAM)
#
#  HOW TO USE:
#  1. Upload LEVIR-CD zip to Kaggle Datasets (run prepare_dataset_for_kaggle.py first)
#  2. Create a new Kaggle Notebook, enable GPU accelerator
#  3. Add your dataset as input
#  4. Paste this file content into the notebook (one script cell)
#  5. Run!
#
#  Dataset path (Kaggle):  /kaggle/input/levir-cd-building-change-detection/LEVIR_CD/
#  Output model saved to:  /kaggle/working/changeformer_levircd_best.pth
# ============================================================


# ============================================================
# CELL 1 -- Install dependencies
# ============================================================
# !pip install -q timm einops torchmetrics


# ============================================================
# CELL 2 -- Imports and Config
# ============================================================
import os
import random
import math
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchmetrics.classification import (
    BinaryF1Score, BinaryJaccardIndex,
    BinaryPrecision, BinaryRecall
)

import timm
import warnings
warnings.filterwarnings("ignore")

# -- Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# -- Paths
# Works both on Kaggle and on a local machine.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("CHANGEFORMER_OUTPUT_DIR", PROJECT_ROOT / "training_output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BEST_MODEL_PATH = OUTPUT_DIR / "changeformer_levircd_best.pth"


def _find_data_root():
    """Auto-detect the LEVIR-CD root folder in either Kaggle or local workspaces."""
    env_root = os.environ.get("LEVIR_CD_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))

    candidates += [
        Path("/kaggle/input/levir-cd"),
        Path("/kaggle/input/levir-cd/LEVIR_CD"),
        Path("/kaggle/input/levir-cd/LEVIR-CD"),
        Path("/kaggle/input/levir-cd/levir_cd"),
        PROJECT_ROOT / "Dataset-Bitemporal" / "LEVIR CD",
        PROJECT_ROOT / "Dataset-Bitemporal" / "LEVIR_CD",
        PROJECT_ROOT / "Dataset-Bitemporal" / "LEVIR-CD",
        PROJECT_ROOT / "Dataset-Bitemporal" / "levir_cd",
        PROJECT_ROOT / "Dataset-Bitemporal",
        PROJECT_ROOT / "Dataset-Zips",
    ]

    for p in candidates:
        if p.exists() and (p / "train" / "A").exists():
            print(f"[DATA] Found LEVIR-CD root: {p}")
            return p

    for p in candidates:
        if p.exists():
            print(f"[WARN] Candidate missing train/A: {p}")
            for item in sorted(p.rglob("*"))[:20]:
                print(f"  {item}")

    raise FileNotFoundError(
        "Could not find LEVIR-CD train/A folder. Set LEVIR_CD_ROOT to your dataset root or "
        "place the dataset under Dataset-Bitemporal/LEVIR CD."
    )

DATA_ROOT = _find_data_root()

# -- Hyper-parameters
CFG = {
    # Dataset
    "patch_size"    : 256,   # tile 1024x1024 into 256x256 patches
    "patch_stride"  : 256,   # non-overlapping for val/test
    "train_stride"  : 128,   # overlapping for train (more diversity)

    # Training
    "num_epochs"    : 60,
    "batch_size"    : 8,     # safe for 16 GB VRAM with MiT-B1
    "lr"            : 6e-5,
    "weight_decay"  : 0.01,
    "warmup_epochs" : 5,
    "num_workers"   : 4,

    # Model
    "backbone"      : "resnet18",  # Compatible timm backbone for local runtime
    "decoder_dim"   : 256,
    "num_classes"   : 1,         # binary: changed vs not-changed

    # Loss
    "bce_weight"    : 0.5,
    "dice_weight"   : 0.5,

    # Device
    "device"        : "cuda" if torch.cuda.is_available() else "cpu",
    "amp"           : True,      # mixed-precision training
}

print(f"Device : {CFG['device']}")
if torch.cuda.is_available():
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM   : {vram:.1f} GB")
print(f"Config : {CFG}")


# ============================================================
# CELL 3 -- LEVIR-CD Dataset
# ============================================================
class LEVIRCDDataset(Dataset):
    """
    LEVIR-CD Building Change Detection Dataset.

    Loads 1024x1024 image pairs and crops patch_size patches on-the-fly.
    A = pre-change image (T1)
    B = post-change image (T2)
    label = binary change mask (255=changed, 0=unchanged)
    """

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, root, split, patch_size=256, stride=256, augment=False):
        super().__init__()
        assert split in ("train", "val", "test")

        self.patch_size = patch_size
        self.stride     = stride
        self.augment    = augment

        split_dir      = root / split
        self.a_dir     = split_dir / "A"
        self.b_dir     = split_dir / "B"
        self.label_dir = split_dir / "label"

        self.image_names = sorted([f.stem for f in self.a_dir.glob("*.png")])
        self.patch_index = self._build_index()
        self.normalize   = T.Normalize(mean=self.MEAN, std=self.STD)

        print(f"  [{split}] {len(self.image_names)} images -> {len(self.patch_index)} patches")

    def _build_index(self):
        n = (1024 - self.patch_size) // self.stride + 1
        index = []
        for name in self.image_names:
            for r in range(n):
                for c in range(n):
                    index.append((name, r, c))
        return index

    def __len__(self):
        return len(self.patch_index)

    def __getitem__(self, idx):
        name, row, col = self.patch_index[idx]
        ps = self.patch_size
        x  = col * self.stride
        y  = row * self.stride

        img_a = Image.open(self.a_dir     / f"{name}.png").convert("RGB")
        img_b = Image.open(self.b_dir     / f"{name}.png").convert("RGB")
        label = Image.open(self.label_dir / f"{name}.png").convert("L")

        img_a = img_a.crop((x, y, x + ps, y + ps))
        img_b = img_b.crop((x, y, x + ps, y + ps))
        label = label.crop((x, y, x + ps, y + ps))

        if self.augment:
            img_a, img_b, label = self._augment(img_a, img_b, label)

        img_a = self.normalize(TF.to_tensor(img_a))
        img_b = self.normalize(TF.to_tensor(img_b))
        label = (torch.from_numpy(np.array(label)).float() > 127).float()

        return img_a, img_b, label

    def _augment(self, a, b, lbl):
        if random.random() > 0.5:
            a, b, lbl = TF.hflip(a), TF.hflip(b), TF.hflip(lbl)
        if random.random() > 0.5:
            a, b, lbl = TF.vflip(a), TF.vflip(b), TF.vflip(lbl)
        k = random.choice([0, 1, 2, 3])
        if k:
            ang = k * 90
            a, b, lbl = TF.rotate(a, ang), TF.rotate(b, ang), TF.rotate(lbl, ang)
        if random.random() > 0.5:
            jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
            a, b = jitter(a), jitter(b)
        return a, b, lbl


# ============================================================
# CELL 4 -- ChangeFormer Model
# ============================================================
def resolve_backbone_name(backbone_name):
    legacy_names = {"mit_b0", "mit_b1", "mit_b2"}
    if backbone_name in legacy_names:
        for fallback in ("resnet18", "resnet34", "efficientnet_b0"):
            try:
                timm.create_model(fallback, pretrained=False)
                return fallback
            except Exception:
                continue
        raise RuntimeError(
            f"Requested legacy backbone '{backbone_name}' is unavailable in this timm build. "
            "Install a compatible timm version or choose an available backbone like 'resnet18'."
        )
    return backbone_name


class MiTEncoder(nn.Module):
    """Shared encoder returning 4-stage feature maps for the siamese decoder."""
    def __init__(self, backbone_name="resnet18", pretrained=True):
        super().__init__()
        backbone_name = resolve_backbone_name(backbone_name)
        out_indices = (1, 2, 3, 4) if backbone_name.startswith("resnet") else (0, 1, 2, 3)
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.out_channels = self.encoder.feature_info.channels()

    def forward(self, x):
        return self.encoder(x)


class ChangeDecoder(nn.Module):
    """
    FPN-style decoder. Takes absolute difference of siamese features
    from all 4 encoder stages, projects to decoder_dim, fuses, predicts.
    """
    def __init__(self, in_channels, decoder_dim=256):
        super().__init__()
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, decoder_dim, 1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True),
            )
            for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(len(in_channels) * decoder_dim, decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_dim // 2, 1, 1),
        )

    def forward(self, feats_a, feats_b, out_size):
        diffs = []
        for fa, fb, proj in zip(feats_a, feats_b, self.proj):
            diff = proj(torch.abs(fa - fb))
            diff = F.interpolate(diff, size=out_size, mode="bilinear", align_corners=False)
            diffs.append(diff)
        x = self.fuse(torch.cat(diffs, dim=1))
        return self.head(x)


class ChangeFormer(nn.Module):
    """
    Siamese ChangeFormer:
      T1 ---> MiT Encoder (shared) ---> feats_A  \
                                                   --> ChangeDecoder --> Change Map
      T2 ---> MiT Encoder (shared) ---> feats_B  /
    """
    def __init__(self, backbone="mit_b1", decoder_dim=256, pretrained=True):
        super().__init__()
        self.encoder = MiTEncoder(backbone, pretrained)
        print(f"  Backbone: {backbone}  |  Channels: {self.encoder.out_channels}")
        self.decoder = ChangeDecoder(self.encoder.out_channels, decoder_dim)

    def forward(self, img_a, img_b):
        H, W = img_a.shape[2], img_a.shape[3]
        feats_a = self.encoder(img_a)
        feats_b = self.encoder(img_b)
        return self.decoder(feats_a, feats_b, out_size=(H, W))  # [B, 1, H, W]


# ============================================================
# CELL 5 -- Loss Function (BCE + Dice)
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        inter = (p * t).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_w=0.5, dice_w=0.5):
        super().__init__()
        self.bce_w    = bce_w
        self.dice_w   = dice_w
        self.bce      = nn.BCEWithLogitsLoss()
        self.dice     = DiceLoss()

    def forward(self, logits, targets):
        t4d = targets.unsqueeze(1)
        return self.bce_w * self.bce(logits, t4d) + self.dice_w * self.dice(logits, t4d)


# ============================================================
# CELL 6 -- LR Scheduler (Linear Warmup + Cosine Decay)
# ============================================================
def build_scheduler(optimizer, num_epochs, warmup_epochs, steps_per_epoch):
    warmup = warmup_epochs * steps_per_epoch
    total  = num_epochs   * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# CELL 7 -- Training Loop
# ============================================================
def train_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    n = len(loader)

    for i, (img_a, img_b, label) in enumerate(loader):
        img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device)
        optimizer.zero_grad()

        with autocast(enabled=CFG["amp"]):
            logits = model(img_a, img_b)
            loss   = criterion(logits, label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if (i + 1) % 50 == 0 or (i + 1) == n:
            lr = scheduler.get_last_lr()[0]
            print(f"  E{epoch+1} [{i+1}/{n}] loss={loss.item():.4f}  lr={lr:.2e}")

    return total_loss / n


# ============================================================
# CELL 8 -- Validation Loop
# ============================================================
@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    f1   = BinaryF1Score(threshold=0.5).to(device)
    iou  = BinaryJaccardIndex(threshold=0.5).to(device)
    prec = BinaryPrecision(threshold=0.5).to(device)
    rec  = BinaryRecall(threshold=0.5).to(device)

    for img_a, img_b, label in loader:
        img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device)
        with autocast(enabled=CFG["amp"]):
            logits = model(img_a, img_b)
            loss   = criterion(logits, label)
        total_loss += loss.item()

        probs = torch.sigmoid(logits.squeeze(1))
        f1.update(probs, label.long())
        iou.update(probs, label.long())
        prec.update(probs, label.long())
        rec.update(probs, label.long())

    return {
        "val_loss" : total_loss / len(loader),
        "f1"       : f1.compute().item(),
        "iou"      : iou.compute().item(),
        "precision": prec.compute().item(),
        "recall"   : rec.compute().item(),
    }


def maybe_save_best_checkpoint(model, optimizer, epoch, metrics, current_best_f1, path=BEST_MODEL_PATH):
    """Save a new best model only when the validation F1 improves."""
    if metrics["f1"] > current_best_f1:
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_f1": metrics["f1"],
            "config": CFG,
        }, path)
        print(f"  >> Best model saved: F1={metrics['f1']:.4f}  path={path}")
        return metrics["f1"]
    return current_best_f1


# ============================================================
# CELL 9 -- Main Training
# ============================================================
def main():
    device = CFG["device"]
    print("\n" + "="*60)
    print("  StatQuery AI -- ChangeFormer on LEVIR-CD")
    print("="*60 + "\n")

    # Datasets
    train_ds = LEVIRCDDataset(DATA_ROOT, "train",
                               patch_size=CFG["patch_size"],
                               stride=CFG["train_stride"], augment=True)
    val_ds   = LEVIRCDDataset(DATA_ROOT, "val",
                               patch_size=CFG["patch_size"],
                               stride=CFG["patch_stride"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                               shuffle=True, num_workers=CFG["num_workers"],
                               pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=CFG["batch_size"] * 2,
                               shuffle=False, num_workers=CFG["num_workers"],
                               pin_memory=True)

    print(f"\nTrain batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

    # Model
    print(f"\nBuilding ChangeFormer ({CFG['backbone']})...")
    model = ChangeFormer(backbone=CFG["backbone"],
                         decoder_dim=CFG["decoder_dim"],
                         pretrained=True).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total/1e6:.2f}M total  |  {trainable/1e6:.2f}M trainable")

    # Optimizer (differential LR: encoder gets 10x lower LR)
    criterion = BCEDiceLoss(bce_w=CFG["bce_weight"], dice_w=CFG["dice_weight"])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": CFG["lr"] * 0.1},
        {"params": model.decoder.parameters(), "lr": CFG["lr"]},
    ], weight_decay=CFG["weight_decay"])

    scheduler = build_scheduler(optimizer, CFG["num_epochs"],
                                CFG["warmup_epochs"], len(train_loader))
    scaler    = GradScaler(enabled=CFG["amp"])

    # Training
    best_f1 = 0.0
    if BEST_MODEL_PATH.exists():
        ckpt = torch.load(BEST_MODEL_PATH, map_location="cpu")
        if "best_f1" in ckpt:
            best_f1 = float(ckpt["best_f1"])
            print(f"  [BEST] Loaded previous best checkpoint: F1={best_f1:.4f} from {BEST_MODEL_PATH}")
    best_epoch = 0
    history   = {"train_loss": [], "val_loss": [], "f1": [], "iou": []}

    print(f"\nStarting {CFG['num_epochs']} epochs...\n")

    for epoch in range(CFG["num_epochs"]):
        print(f"\n-- Epoch {epoch+1}/{CFG['num_epochs']} " + "-"*40)

        tr_loss = train_epoch(model, train_loader, optimizer, criterion,
                              scaler, scheduler, device, epoch)
        metrics = validate(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["f1"].append(metrics["f1"])
        history["iou"].append(metrics["iou"])

        print(f"\n  Summary  |  TrainLoss: {tr_loss:.4f}  ValLoss: {metrics['val_loss']:.4f}")
        print(f"           |  F1: {metrics['f1']:.4f}  IoU: {metrics['iou']:.4f}")
        print(f"           |  Precision: {metrics['precision']:.4f}  Recall: {metrics['recall']:.4f}")

        # Save best only when the new metric is strictly better
        if metrics["f1"] > best_f1:
            best_f1 = maybe_save_best_checkpoint(model, optimizer, epoch, metrics, best_f1)
            best_epoch = epoch + 1
        else:
            print(f"  >> Keeping previous best checkpoint: F1={best_f1:.4f}  path={BEST_MODEL_PATH}")

        # Checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt = OUTPUT_DIR / f"changeformer_epoch_{epoch+1}.pth"
            torch.save({"epoch": epoch+1, "model_state": model.state_dict(),
                        "f1": metrics["f1"], "config": CFG}, ckpt)
            print(f"  >> Checkpoint saved: {ckpt}")

    print(f"\n{'='*60}")
    print(f"  Training Complete!  Best F1={best_f1:.4f} at epoch {best_epoch}")
    print(f"{'='*60}\n")

    plot_history(history)
    return model, history


# ============================================================
# CELL 10 -- Plots and Visualization
# ============================================================
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(ep, history["train_loss"], label="Train", color="#4A90D9", lw=2)
    axes[0].plot(ep, history["val_loss"],   label="Val",   color="#E74C3C", lw=2)
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history["f1"],  label="F1",  color="#2ECC71", lw=2)
    axes[1].plot(ep, history["iou"], label="IoU", color="#9B59B6", lw=2)
    axes[1].set(title="Metrics", xlabel="Epoch", ylabel="Score", ylim=[0, 1])
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Training curves saved: {out}")


@torch.no_grad()
def visualize_predictions(model, dataset, device, n=4):
    model.eval()
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    idxs = random.sample(range(len(dataset)), min(n, len(dataset)))
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))

    for row, idx in enumerate(idxs):
        img_a, img_b, label = dataset[idx]
        a_disp = (img_a * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()
        b_disp = (img_b * STD + MEAN).clamp(0, 1).permute(1, 2, 0).numpy()

        logit    = model(img_a.unsqueeze(0).to(device), img_b.unsqueeze(0).to(device))
        pred_bin = (torch.sigmoid(logit).squeeze().cpu().numpy() > 0.5).astype(float)

        axes[row, 0].imshow(a_disp);              axes[row, 0].set_title("T1 (Pre)")
        axes[row, 1].imshow(b_disp);              axes[row, 1].set_title("T2 (Post)")
        axes[row, 2].imshow(label, cmap="gray");  axes[row, 2].set_title("Ground Truth")
        axes[row, 3].imshow(pred_bin, cmap="gray"); axes[row, 3].set_title("Prediction")
        for ax in axes[row]: ax.axis("off")

    plt.suptitle("ChangeFormer Predictions on LEVIR-CD", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = OUTPUT_DIR / "predictions_visualization.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Predictions saved: {out}")


# ============================================================
# CELL 11 -- Run
# ============================================================
if __name__ == "__main__":
    model, history = main()

    print("\nGenerating visualizations...")
    best_ckpt = torch.load(BEST_MODEL_PATH, map_location=CFG["device"])
    model.load_state_dict(best_ckpt["model_state"])

    val_vis = LEVIRCDDataset(DATA_ROOT, "val",
                              patch_size=CFG["patch_size"],
                              stride=CFG["patch_stride"], augment=False)
    visualize_predictions(model, val_vis, CFG["device"], n=4)

    print("\nDone! Download from Kaggle Output tab:")
    print("  changeformer_levircd_best.pth    <- best model weights")
    print("  training_curves.png              <- loss/metric plots")
    print("  predictions_visualization.png    <- visual results")
