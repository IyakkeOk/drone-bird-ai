import os
import yaml
import torch
import timm
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, classification_report

from classification.datasets.cub200 import CUB200Dataset


def load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


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
    # Dataset
    # -----------------------------
    train_ds = CUB200Dataset(
        root=cfg["data_root"],
        train=True,
        transform=train_tfms
    )

    val_ds = CUB200Dataset(
        root=cfg["data_root"],
        train=False,
        transform=val_tfms
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    # -----------------------------
    # Model: MobileOne-S1
    # -----------------------------
    model = timm.create_model(
        "mobileone_s1",
        pretrained=True,
        num_classes=200
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    best_acc = 0.0

    # -----------------------------
    # Training Loop
    # -----------------------------
    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device)

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

        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "mobileone_s1_cub_best.pth")

    print("\nFinal Classification Report:")
    print(classification_report(gts, preds, digits=4))


if __name__ == "__main__":
    main()
