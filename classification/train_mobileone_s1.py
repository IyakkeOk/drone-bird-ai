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


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    """Plot and save a confusion matrix heatmap."""
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title("Full Confusion Matrix")
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_distribution(errors, save_path="error_distribution.png"):
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
        # zero out diagonal (correct predictions)
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
    summary = {
        "Mean class recall": np.mean(recalls),
        "Median class recall": np.median(recalls),
        "Worst class recall": np.min(recalls),
        "Classes with recall < 90%": np.sum(recalls < 90)
    }
    return summary


def plot_per_class_f1(f1_scores, save_path="per_class_f1.png"):
    """Plot bar chart of per-class F1 scores."""
    plt.figure(figsize=(16, 6))
    plt.bar(range(len(f1_scores)), f1_scores)
    plt.xlabel("Class Index")
    plt.ylabel("F1 Score (%)")
    plt.title("Per-Class F1 Scores")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_loss_curves(train_losses, val_accs, save_path_loss="training_loss.png", save_path_acc="validation_acc.png"):
    """Plot training loss and validation accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    plt.figure()
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path_loss, dpi=300)
    plt.close()

    plt.figure()
    plt.plot(epochs, val_accs, marker='o', color='green', label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy Curve")
    plt.grid(True)
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

    print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # -----------------------------
    # Model: MobileOne-S1
    # -----------------------------
    model = timm.create_model("mobileone_s1", pretrained=True, num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_acc = 0.0
    train_losses = []
    val_accs = []

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
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
                outputs = model(images)
                pred = torch.argmax(outputs, dim=1).cpu().numpy()
                preds.extend(pred)
                gts.extend(labels.numpy())

        acc = accuracy_score(gts, preds)
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accs.append(acc)

        print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "mobileone_s1_cub_best.pth")

    # -----------------------------
    # Final Metrics & PhD-level Analysis
    # -----------------------------
    print("\nFinal Classification Report:")
    print(classification_report(gts, preds, digits=4))

    # Confusion matrix
    cm = confusion_matrix(gts, preds)
    class_names = [f"Class_{i}" for i in range(200)]

    # Save full confusion matrix as CSV
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv("confusion_matrix_full.csv")

    # Heatmap of confusion matrix
    plot_confusion_matrix(cm, class_names, save_path="confusion_matrix_heatmap.png")

    # Error distribution
    errors = cm.sum(axis=1) - np.diag(cm)
    plot_error_distribution(errors, save_path="error_distribution.png")

    # Top-3 misclassifications per class
    top3_miscls = top_k_misclassifications(cm, class_names, k=3)
    top3_df = pd.DataFrame.from_dict(top3_miscls, orient='index')
    top3_df.to_csv("top3_misclassifications.csv")
    print("\nTop-3 misclassifications saved to top3_misclassifications.csv")

    # Statistical summary
    summary = compute_statistical_summary(cm)
    print("\nStatistical Summary of Class-level Performance:")
    for k, v in summary.items():
        print(f"{k}: {v:.2f}%")

    # -----------------------------
    # Top-K Accuracy
    # -----------------------------
    y_true = np.array(gts)
    y_pred_probs = np.zeros((len(y_true), 200))
    model.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            y_pred_probs[labels.numpy(), :] = probs  # Fill for Top-K computation

    top1 = top_k_accuracy_score(y_true, y_pred_probs, k=1)
    top3 = top_k_accuracy_score(y_true, y_pred_probs, k=3)
    top5 = top_k_accuracy_score(y_true, y_pred_probs, k=5)
    print(f"\nTop-1 Accuracy: {top1 * 100:.2f}%")
    print(f"Top-3 Accuracy: {top3 * 100:.2f}%")
    print(f"Top-5 Accuracy: {top5 * 100:.2f}%")

    # -----------------------------
    # Per-class F1 Score Plot
    # -----------------------------
    f1_scores = f1_score(gts, preds, average=None) * 100
    plot_per_class_f1(f1_scores, save_path="per_class_f1.png")

    # -----------------------------
    # Training & Validation Curves
    # -----------------------------
    plot_loss_curves(train_losses, val_accs, save_path_loss="training_loss.png", save_path_acc="validation_acc.png")

    print("\nAll metrics, Top-K, F1 plot, and curves saved to working directory.")


if __name__ == "__main__":
    main()
