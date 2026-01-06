import os
import yaml
import torch
import timm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import sys

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .datasets.cub200 import CUB200Dataset


def load_config(cfg_path):
    """Load YAML configuration file."""
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Results directory setup (MODEL-SPECIFIC)
# --------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_ghostnet_v2")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

for d in [RESULTS_DIR, CKPT_DIR, METRICS_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"[INFO] GhostNet-V2 results will be saved to: {RESULTS_DIR}")


# --------------------------------------------------
# Visualization Utilities (UNCHANGED)
# --------------------------------------------------
def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Full Confusion Matrix (GhostNet-V2)")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_distribution(errors, save_path):
    """Plot histogram of number of misclassified samples per class."""
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(errors)), errors)
    plt.xlabel("Class Index")
    plt.ylabel("Number of Misclassifications")
    plt.title("Error Distribution per Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_per_class_f1(f1_scores, save_path):
    """Plot bar chart of per-class F1 scores."""
    plt.figure(figsize=(16, 6))
    plt.bar(range(len(f1_scores)), f1_scores)
    plt.xlabel("Class Index")
    plt.ylabel("F1 Score (%)")
    plt.title("Per-Class F1 Scores")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curves(train_losses, val_accs, save_path_loss, save_path_acc):
    """Plot training loss and validation accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.savefig(save_path_loss, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_accs, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.legend()
    plt.savefig(save_path_acc, dpi=300)
    plt.close()


# --------------------------------------------------
# Statistical Summary (UNCHANGED, PhD-GRADE)
# --------------------------------------------------
def compute_statistical_summary(cm):
    """
    Compute macro-level statistical summaries for multi-class classification.
    Includes precision, recall, F1-score, and overall accuracy.
    """
    num_classes = cm.shape[0]
    precisions, recalls, f1s = [], [], []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision * 100)
        recalls.append(recall * 100)
        f1s.append(f1 * 100)

    overall_accuracy = np.trace(cm) / np.sum(cm) * 100

    return {
        "Overall accuracy (%)": float(overall_accuracy),

        "Mean class precision (%)": float(np.mean(precisions)),
        "Median class precision (%)": float(np.median(precisions)),
        "Worst class precision (%)": float(np.min(precisions)),

        "Mean class recall (%)": float(np.mean(recalls)),
        "Median class recall (%)": float(np.median(recalls)),
        "Worst class recall (%)": float(np.min(recalls)),

        "Mean class F1-score (%)": float(np.mean(f1s)),
        "Median class F1-score (%)": float(np.median(f1s)),
        "Worst class F1-score (%)": float(np.min(f1s)),

        "Classes with recall < 90%": int(np.sum(np.array(recalls) < 90)),
        "Classes with F1-score < 90%": int(np.sum(np.array(f1s) < 90)),
    }


# --------------------------------------------------
# Main Training & Evaluation Pipeline
# --------------------------------------------------
def main():
    cfg = load_config("../configs/cub_ghostnet_v2.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Data transforms
    # -----------------------------
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    # -----------------------------
    # Datasets & Loaders
    # -----------------------------
    train_ds = CUB200Dataset(cfg["data_root"], train=True, transform=train_tfms)
    val_ds = CUB200Dataset(cfg["data_root"], train=False, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=4)

    # -----------------------------
    # Model: GhostNet-V2
    # -----------------------------
    model = timm.create_model(
        "ghostnetv2_100",
        pretrained=True,
        num_classes=200
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    best_acc = 0.0
    train_losses, val_accs = [], []
    epoch_log = []

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        preds, gts = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                pred = torch.argmax(model(images), dim=1).cpu().numpy()
                preds.extend(pred)
                gts.extend(labels.numpy())

        acc = accuracy_score(gts, preds)
        avg_loss = running_loss / len(train_loader)

        train_losses.append(avg_loss)
        val_accs.append(acc)

        epoch_log.append({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "val_accuracy": acc
        })

        pd.DataFrame(epoch_log).to_csv(
            os.path.join(METRICS_DIR, "epoch_metrics.csv"), index=False
        )

        torch.save(
            model.state_dict(),
            os.path.join(CKPT_DIR, f"epoch_{epoch+1:03d}.pth")
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(
                model.state_dict(),
                os.path.join(CKPT_DIR, "best_model.pth")
            )

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

    # -----------------------------
    # Final Metrics & Analysis
    # -----------------------------
    cm = confusion_matrix(gts, preds)
    class_names = [f"Class_{i}" for i in range(200)]

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        os.path.join(TABLES_DIR, "confusion_matrix_full.csv")
    )

    plot_confusion_matrix(cm, class_names,
                          os.path.join(PLOTS_DIR, "confusion_matrix_heatmap.png"))

    errors = cm.sum(axis=1) - np.diag(cm)
    plot_error_distribution(errors,
                            os.path.join(PLOTS_DIR, "error_distribution.png"))

    f1_scores = f1_score(gts, preds, average=None) * 100
    plot_per_class_f1(f1_scores,
                      os.path.join(PLOTS_DIR, "per_class_f1.png"))

    plot_loss_curves(train_losses, val_accs,
                     os.path.join(PLOTS_DIR, "training_loss.png"),
                     os.path.join(PLOTS_DIR, "validation_acc.png"))

    summary = compute_statistical_summary(cm)
    with open(os.path.join(METRICS_DIR, "final_summary.yaml"), "w") as f:
        yaml.dump(summary, f)

    print("\nFinal Statistical Summary (GhostNet-V2):")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
