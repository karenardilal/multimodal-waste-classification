"""
Multimodal waste classification (Image + filename text)

This script trains and evaluates a multimodal classifier for 4-class waste sorting: Black, Blue, Green, TTR

Inputs:
  (1) Image: RGB photo of the waste item
  (2) Text: filename-derived description (e.g., "food metal can") encoded with TF-IDF

High-level pipeline:
  1) Load Train/Val/Test splits using ImageFolder
  2) Extract short text from filenames and fit a TF-IDF vectorizer on TRAIN only (avoid leakage)
  3) Build a multimodal Dataset that returns (image, tfidf_vector, label, path, raw_text)
  4) Train a model:
       - Image branch: pretrained ResNet50 -> 2048-d features
       - Text branch: small MLP -> txt_hidden embedding
       - Fusion: concatenate image + text features -> classifier
  5) Two-stage training:
       - Warm-up: train text + classifier only (freeze ResNet)
       - Fine-tune: unfreeze ResNet layer3 + layer4 with discriminative learning rates
  6) Track metrics (Acc, BalancedAcc, MacroF1), save best model, and export figures on test:
       - confusion matrix
       - misclassified image grid

Cluster stability notes (TALC):
  - Set MKL/OpenMP env vars BEFORE importing numpy/torch to avoid loader crashes
  - Use DataLoader num_workers=0 (no multiprocessing)
"""


# MUST BE FIRST: avoid MKL/OpenMP loader crashes
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

import time
import copy
import re
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (confusion_matrix, classification_report, balanced_accuracy_score, f1_score)


# Reproducibility / randomness control
def seed_everything(seed: int = 42):
    """
    Set random seeds to make runs more reproducible
    Notes:
      - We seed python, numpy, and torch (CPU/GPU)
      - cudnn.benchmark=True can improve performance but may reduce strict determinism
        For this assignment, we prioritized speed/stability on TALC while keeping seeding
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic=False and benchmark=True is a common tradeoff
    # - faster convolutions (benchmark picks best algo)
    # - slightly less strict repeatability between runs
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)


# Paths / output locations
data_dir = r"/work/TALC/ensf617_2026w/garbage_data/CVPR_2024_dataset_"
train_dir = os.path.join(data_dir + "Train")
val_dir   = os.path.join(data_dir + "Val")
test_dir  = os.path.join(data_dir + "Test")

# Where to save best model weights (based on validation metric)
save_path = "best_model_multimodal.pth"

# Folder for figures (confusion matrix + misclassified grid)
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

# Image transforms (augmentation for train, deterministic preprocess for val/test)
# ResNet pretrained weights expect ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = {
    "train": transforms.Compose([
        # Stronger augmentation for better generalization
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),

        # Ensure RGB consistently (some PIL images could be L/LA/etc.)
        transforms.Lambda(lambda img: img.convert("RGB")),

        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),

        # RandomErasing is applied after normalization; acts like occlusion augmentation
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
    ]),
    "val": transforms.Compose([
        # Standard deterministic preprocessing for evaluation
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    "test": transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
}


# Helper: convert image filepath -> short text string
def filename_to_text(path: str) -> str:
    """
    Convert a filename into a short text description to feed TF-IDF.

    Example:
      'food_metal_can_2129.png' -> 'food metal can'

    Steps:
      - Remove extension
      - Replace underscores with spaces
      - Lowercase
      - Remove digits (IDs / indices)
      - Collapse multiple spaces
    """
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    txt = stem.replace("_", " ").lower()
    txt = re.sub(r"\d+", " ", txt)          # remove digits
    txt = re.sub(r"\s+", " ", txt).strip()  # normalize whitespace
    return txt

# Load datasets (ImageFolder expects directory structure: split/class_name/*.png)
train_img = datasets.ImageFolder(train_dir, transform=transform["train"])
val_img   = datasets.ImageFolder(val_dir,   transform=transform["val"])
test_img  = datasets.ImageFolder(test_dir,  transform=transform["test"])

class_names = train_img.classes
num_classes = len(class_names)

def print_split_distribution(name, dataset: datasets.ImageFolder):
    """
    Print how many samples per class are in a split.
    Useful sanity check for imbalance and to confirm split loads correctly.
    """
    targets = np.array(dataset.targets)
    counts = Counter(targets)
    print(f"\n[{name}] samples: {len(dataset)}")
    for c in range(num_classes):
        print(f"  {class_names[c]:<8s}: {counts.get(c, 0)}")
    return counts

_ = print_split_distribution("TRAIN", train_img)
_ = print_split_distribution("VAL",   val_img)
_ = print_split_distribution("TEST",  test_img)


# TF-IDF features from filenames (fit on TRAIN only to avoid text leakage)
train_texts = [filename_to_text(p) for (p, _) in train_img.samples]
val_texts   = [filename_to_text(p) for (p, _) in val_img.samples]
test_texts  = [filename_to_text(p) for (p, _) in test_img.samples]

# Filenames are short, we use a compact vocabulary for efficiency
vectorizer = TfidfVectorizer(
    lowercase=True,
    token_pattern=r"(?u)\b[a-zA-Z]+\b",  # keep alphabetic tokens only
    ngram_range=(1, 2),                  # unigram + bigram features
    max_features=2000                    # cap size to control memory/model size
)

# Fit TF-IDF on training filenames only, val/test are transformed using same vocab
X_train_txt = vectorizer.fit_transform(train_texts)
X_val_txt   = vectorizer.transform(val_texts)
X_test_txt  = vectorizer.transform(test_texts)

txt_dim = X_train_txt.shape[1]
print(f"\nTF-IDF vocab size (features): {txt_dim}")

# Multimodal Dataset: aligns ImageFolder samples with TF-IDF rows
class MultiModalDataset(Dataset):
    """
    Dataset that returns paired (image, text_vector, label) samples.

    We wrap an ImageFolder dataset and a TF-IDF matrix that is aligned with
    the ImageFolder .samples list order.

    Returns per item:
      - img:      torch.FloatTensor [3,224,224] (already transformed + normalized)
      - x_txt:    torch.FloatTensor [txt_dim] (dense TF-IDF vector)
      - label:    torch.LongTensor scalar
      - path:     original filepath (useful for debugging/inspection)
      - raw_text: original filename-derived text (used in misclassified grid titles)
    """
    def __init__(self, imagefolder: datasets.ImageFolder, tfidf_matrix, raw_texts):
        self.ds = imagefolder
        self.X = tfidf_matrix
        self.raw_texts = raw_texts

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # ImageFolder returns (image_tensor, class_index)
        img, label = self.ds[idx]
        path, _ = self.ds.samples[idx]

        # TF-IDF is stored as sparse; convert one row to dense float32
        x_txt = self.X[idx].toarray().astype(np.float32).squeeze(0)
        x_txt = torch.from_numpy(x_txt)

        raw = self.raw_texts[idx]
        return img, x_txt, torch.tensor(label, dtype=torch.long), path, raw

train_dataset = MultiModalDataset(train_img, X_train_txt, train_texts)
val_dataset   = MultiModalDataset(val_img,   X_val_txt,   val_texts)
test_dataset  = MultiModalDataset(test_img,  X_test_txt,  test_texts)

# DataLoaders (SAFE MODE for TALC)
num_workers = 2
pin_memory = True

# WeightedRandomSampler addresses class imbalance by oversampling minority classes
train_targets = np.array(train_img.targets)
class_counts = Counter(train_targets)

# Inverse-frequency weights (rare classes get larger weight)
class_weights = {c: 1.0 / class_counts[c] for c in range(num_classes)}
sample_weights = np.array([class_weights[t] for t in train_targets], dtype=np.float64)

sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights), num_samples=len(sample_weights), replacement=True)

def multimodal_collate(batch):
    """
    Custom collate function because each item includes both tensors + metadata.
    Stacks images and text vectors into batch tensors, keeps paths/raw strings as lists.
    """
    imgs, txts, labels, paths, raws = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    txts = torch.stack(txts, dim=0)
    labels = torch.stack(labels, dim=0)
    return imgs, txts, labels, list(paths), list(raws)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=sampler,          # sampler implies shuffle=False, both can't be used at the same time
    shuffle=False,            # WeightedRandomSampler already randomizes the sampling for the training set!!
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=multimodal_collate,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=multimodal_collate,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    collate_fn=multimodal_collate,
)

dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}


# Device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))


# Multimodal model definition
class MultiModalResNetTFIDF(nn.Module):
    """
    Two-branch multimodal model:
      - ResNet50 backbone produces a 2048-d image embedding
      - Text MLP produces a txt_hidden-d embedding from TF-IDF
      - Fusion is concatenation: [img_feat | txt_feat]
      - Classifier predicts one of num_classes labels
    """
    def __init__(self, num_classes: int, txt_dim: int, txt_hidden: int = 256, dropout_p: float = 0.3):
        super().__init__()

        # Image encoder (pretrained ResNet50)
        weights = ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)

        # ResNet "fc" is the original ImageNet classifier, replace with Identity
        # so the backbone outputs the feature vector directly
        in_feats = self.backbone.fc.in_features  # typically 2048 for ResNet50
        self.backbone.fc = nn.Identity()

        # Text encoder (TF-IDF -> embedding)
        self.txt_mlp = nn.Sequential(
            nn.Linear(txt_dim, txt_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_p),
        )

        # Fusion classifier
        fusion_dim = in_feats + txt_hidden
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, images, text_vec):
        """
        Forward pass
        Inputs:
          - images:   [B,3,224,224]
          - text_vec: [B,txt_dim]
        Output:
          - logits:   [B,num_classes]
        """
        img_feat = self.backbone(images)      # [B,2048]
        txt_feat = self.txt_mlp(text_vec)     # [B,txt_hidden]
        fused = torch.cat([img_feat, txt_feat], dim=1)
        logits = self.classifier(fused)
        return logits

# Instantiate model
model = MultiModalResNetTFIDF(
    num_classes=num_classes,
    txt_dim=txt_dim,
    txt_hidden=256,
    dropout_p=0.3
)

# Warm-up strategy: freeze ResNet initially (only learn text + classifier head)
for p in model.backbone.parameters():
    p.requires_grad = False

# Ensure text and classifier are trainable
for p in model.txt_mlp.parameters():
    p.requires_grad = True
for p in model.classifier.parameters():
    p.requires_grad = True

model = model.to(device)

# Loss function
# Label smoothing can reduce overconfidence and sometimes improves generalization
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# Optimizers: warm-up then discriminative fine-tuning
def make_optimizer_warmup(model_):
    """
    Warm-up optimizer:
      - Only trains text MLP + classifier parameters (ResNet frozen)
      - AdamW uses decoupled weight decay (often better behaved than Adam + L2)
    """
    params = list(model_.txt_mlp.parameters()) + list(model_.classifier.parameters())
    return optim.AdamW(params, lr=3e-4, weight_decay=1e-2)

def unfreeze_layer3_layer4(model_):
    """
    Unfreeze later ResNet blocks for fine-tuning
    layer3 and layer4 contain higher-level features that adapt well to the new dataset
    without fully retraining early low-level filters
    """
    for p in model_.backbone.layer3.parameters():
        p.requires_grad = True
    for p in model_.backbone.layer4.parameters():
        p.requires_grad = True

def make_optimizer_discriminative(model_):
    """
    Fine-tuning optimizer with discriminative learning rates:
      - smaller LR for backbone layers (avoid destroying pretrained features)
      - larger LR for text MLP and classifier (learn task-specific mapping)
    """
    return optim.AdamW(
        [
            {"params": model_.backbone.layer3.parameters(), "lr": 5e-5},
            {"params": model_.backbone.layer4.parameters(), "lr": 1e-4},
            {"params": model_.txt_mlp.parameters(),         "lr": 3e-4},
            {"params": model_.classifier.parameters(),      "lr": 3e-4},
        ],
        weight_decay=1e-2
    )

# Plotting and saving confusion matrix used for final evaluation reporting
def save_confusion_matrix(cm, title, save_file, class_names_):
    fig = plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names_))
    plt.xticks(tick_marks, class_names_, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names_)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Add numeric labels on top of the heatmap
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.tight_layout()
    fig.savefig(save_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

# Reverses ImageNet normalization so the images can be displayed correctly
def denormalize(img_tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    x = img_tensor.cpu() * std + mean # img_tensor [3,H,W] tensor that was normalized by ImageNet mean/std
    x = torch.clamp(x, 0.0, 1.0)
    return x.permute(1, 2, 0).numpy() # numpy image [H,W,3] in [0,1], suitable for matplotlib imshow()

# Grid with figures of the incorrect classifications
def save_misclassified_grid(mis_items, save_file, max_items=32):
    k = min(len(mis_items), max_items)
    if k == 0:
        print("No misclassifications to plot.")
        return

    cols = 8
    rows = int(np.ceil(k / cols))
    fig = plt.figure(figsize=(cols * 2.2, rows * 2.6))

    for idx in range(k):
        item = mis_items[idx]
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(denormalize(item["img"]))
        ax.axis("off")
        ax.set_title(
            f"T:{item['true_name']}\nP:{item['pred_name']}\n{item['raw_text']}",
            fontsize=7
        )

    plt.tight_layout()
    fig.savefig(save_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


# Evaluation function (VAL/TEST metrics + optional figure saving)
@torch.no_grad()
def evaluate(model_, dataloader_, split_name="VAL", save_figs=False):
    """
    Run inference on a dataloader and compute:
      - loss
      - accuracy
      - balanced accuracy
      - macro F1
      - classification report
      - confusion matrix

    If save_figs=True, also saves:
      - confusion matrix plot
      - misclassified examples grid (up to 32)
    """
    model_.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0
    n = len(dataloader_.dataset)

    # Only filled if save_figs=True (used for qualitative inspection)
    mis_items = []

    for imgs, txts, labels, paths, raws in dataloader_:
        imgs = imgs.to(device, non_blocking=True)
        txts = txts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model_(imgs, txts)
        loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)

        running_loss += loss.item() * imgs.size(0)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        if save_figs:
            # Collect misclassified samples for visualization
            mism = (preds != labels).detach().cpu().numpy()
            for i, is_mis in enumerate(mism):
                if is_mis:
                    true_i = int(labels[i].detach().cpu().item())
                    pred_i = int(preds[i].detach().cpu().item())
                    mis_items.append({
                        "img": imgs[i].detach().cpu(),
                        "true_name": class_names[true_i],
                        "pred_name": class_names[pred_i],
                        "raw_text": raws[i],
                    })

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    avg_loss = running_loss / n
    acc = (all_preds == all_labels).mean()
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"\n[{split_name}] Loss: {avg_loss:.4f}  Acc: {acc:.4f}  BalancedAcc: {bal_acc:.4f}  MacroF1: {macro_f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:\n", cm)

    if save_figs:
        cm_path = os.path.join(out_dir, f"confusion_matrix_{split_name}.png")
        save_confusion_matrix(cm, f"Confusion Matrix ({split_name})", cm_path, class_names)

        grid_path = os.path.join(out_dir, f"misclassified_{split_name}_grid.png")
        save_misclassified_grid(mis_items, grid_path, max_items=32)

        print(f"\nSaved confusion matrix: {cm_path}")
        print(f"Saved misclassified grid: {grid_path}")

    return {"loss": avg_loss, "acc": acc, "balanced_acc": bal_acc, "macro_f1": macro_f1}


# Training loop (warm-up + fine-tuning + early stopping)
def train_model(model_, dataloaders_, num_epochs_total=12, warmup_epochs=2, early_stop_patience=4, monitor_metric="balanced_acc"):
    """
    Train the model with:
      - Warm-up stage (freeze backbone)
      - Fine-tuning stage (unfreeze layer3+layer4)
      - ReduceLROnPlateau scheduler (on validation metric)
      - Early stopping based on monitor_metric
      - Save best model weights to save_path

    Returns:
      model_ loaded with best checkpoint weights
    """
    best_state = copy.deepcopy(model_.state_dict())
    best_metric = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    # Warm-up optimizer (text + classifier only)
    optimizer = make_optimizer_warmup(model_)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    for epoch in range(num_epochs_total):
        print(f"\nEpoch {epoch + 1}/{num_epochs_total}")
        t0 = time.time()

        # Switch to fine-tuning after warmup_epochs
        if epoch == warmup_epochs:
            print("\nSwitching to fine-tuning: unfreeze ResNet layer3 + layer4 (discriminative LR)")
            unfreeze_layer3_layer4(model_)
            optimizer = make_optimizer_discriminative(model_)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=2
            )

        model_.train()
        running_loss = 0.0
        running_correct = 0
        n_train = len(dataloaders_["train"].dataset)

        # One training epoch
        for imgs, txts, labels, _, _ in dataloaders_["train"]:
            imgs = imgs.to(device, non_blocking=True)
            txts = txts.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model_(imgs, txts)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            running_correct += (preds == labels).sum().item()

        print(f"[TRAIN] Loss: {running_loss / n_train:.4f}  Acc: {running_correct / n_train:.4f}")

        # Validation + scheduling
        val_metrics = evaluate(model_, dataloaders_["val"], split_name="VAL", save_figs=False)
        scheduler.step(val_metrics[monitor_metric])

        current_lrs = [pg["lr"] for pg in optimizer.param_groups]
        print(f"Current LRs: {current_lrs}")
        print(f"Epoch time: {time.time() - t0:.1f}s")

        # Early stopping + checkpointing
        current_metric = val_metrics[monitor_metric]
        if current_metric > best_metric + 1e-6:
            best_metric = current_metric
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model_.state_dict())

            # Save best checkpoint so we can reload for final test evaluation
            torch.save(best_state, save_path)

            epochs_no_improve = 0
            print(f"New best {monitor_metric}: {best_metric:.4f} (epoch {best_epoch}) saved -> {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement. Patience: {epochs_no_improve}/{early_stop_patience}")

        if epochs_no_improve >= early_stop_patience:
            print(f"Early stopping. Best {monitor_metric}: {best_metric:.4f} at epoch {best_epoch}.")
            break

    # Load best checkpoint weights into the returned model
    model_.load_state_dict(best_state)
    print(f"\nBest epoch: {best_epoch} | Best {monitor_metric}: {best_metric:.4f}")
    return model_


# Run training + final test evaluation
model = train_model(model, dataloaders)

print("\n=========================")
print("FINAL TEST EVALUATION")
print("=========================")

# Load best saved checkpoint
model.load_state_dict(torch.load(save_path, map_location=device))

# For TEST, we save figures for the report
test_metrics = evaluate(model, dataloaders["test"], split_name="TEST", save_figs=True)
print("Test Macro F1:", test_metrics["macro_f1"])
print(f"\nFigures saved under: {out_dir}/")