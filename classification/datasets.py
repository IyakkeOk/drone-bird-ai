import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CUB200Dataset(Dataset):
    """
    Custom Dataset for CUB-200-2011
    Handles official train/test split.
    """

    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

        # Load metadata
        self.image_paths = {}
        self.image_labels = {}
        self.split = {}

        with open(os.path.join(root_dir, "images.txt")) as f:
            for line in f:
                idx, path = line.strip().split()
                self.image_paths[int(idx)] = path

        with open(os.path.join(root_dir, "image_class_labels.txt")) as f:
            for line in f:
                idx, label = line.strip().split()
                self.image_labels[int(idx)] = int(label) - 1  # 0-based

        with open(os.path.join(root_dir, "train_test_split.txt")) as f:
            for line in f:
                idx, is_train = line.strip().split()
                self.split[int(idx)] = int(is_train)

        self.samples = [
            (self.image_paths[i], self.image_labels[i])
            for i in self.image_paths
            if self.split[i] == int(train)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, "images", img_rel_path)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label
