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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, top_k_accuracy_score, f1_score

import sys

# --------------------------------------------------
# Ensure project root is on PYTHONPATH
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from classification.datasets.cub200 import CUB200Dataset


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


# --------------------------------------------------
# Results directory setup (ADDED)
# --------------------------------------------------
RESULTS_DIR = "results"
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")

for d in [RESULTS_DIR, CKPT_DIR, METRICS_DIR, PLOTS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Full Confusion Matrix")
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


def top_k_misclassifications(cm, class_names, k=3):
    """Compute top-K misclassified classes per true class."""
    top_k = {}
    for i, row in enumerate(cm):
        misclass_row = row.copy()
        misclass_row[i] = 0
        top_indices = misclass_row.argsort()[::-1][:k]
        top_k[class_names[i]] = [(class_names[j], misclass_row[j]) for j in top_indices if misclass_row[j] > 0]
    return top_k


def compute_statistical_summary(cm):
    """Compute mean, median, worst class recall and count of classes < 90% recall."""
    recalls = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall * 100)
    recalls = np.array(recalls)
    return {
        "Mean class recall (%)": float(np.mean(recalls)),
        "Median class recall (%)": float(np.median(recalls)),
        "Worst class recall (%)": float(np.min(recalls)),
        "Classes with recall < 90%": int(np.sum(recalls < 90))
    }


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
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(False)
    plt.legend()
    plt.savefig(save_path_loss, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_accs, marker='o', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.grid(False)
    plt.legend()
    plt.savefig(save_path_acc, dpi=300)
    plt.close()


def main():
    cfg = load_config("../configs/cub_mobileone_s1.yaml")
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
    # Datasets
    # -----------------------------
    train_ds = CUB200Dataset(root=cfg["data_root"], train=True, transform=train_tfms)
    val_ds = CUB200Dataset(root=cfg["data_root"], train=False, transform=val_tfms)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

    # -----------------------------
    # Model: MobileOne-S1
    # -----------------------------
    model = timm.create_model("mobileone_s1", pretrained=True, num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

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


if __name__ == "__main__":
    main()
