import os
from PIL import Image
from torch.utils.data import Dataset


class CUB200Dataset(Dataset):
    """
    Official CUB-200-2011 Dataset Loader

    Folder structure expected:
    root/
      CUB_200_2011/
        images/
        images.txt
        image_class_labels.txt
        train_test_split.txt
    """

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        cub_root = os.path.join(root, "CUB_200_2011")

        self.image_dir = os.path.join(cub_root, "images")

        # Read metadata files
        self.images = self._read_file(os.path.join(cub_root, "images.txt"))
        self.labels = self._read_file(os.path.join(cub_root, "image_class_labels.txt"))
        self.splits = self._read_file(os.path.join(cub_root, "train_test_split.txt"))

        self.samples = []

        for img_id in self.images:
            is_train = self.splits[img_id] == 1
            if is_train == self.train:
                img_path = self.images[img_id]
                label = self.labels[img_id] - 1  # 0-based
                self.samples.append((img_path, label))

    def _read_file(self, path):
        data = {}
        with open(path, "r") as f:
            for line in f:
                key, value = line.strip().split()
                data[int(key)] = int(value) if value.isdigit() else value
        return data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_rel_path, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_rel_path)

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
