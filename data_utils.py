import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageMaskLabelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_labels = self._load_labels(label_file)

    def _load_labels(self, label_file):
        image_labels = {}
        with open(label_file, 'r') as f:
            for line in f:
                image_name, label = line.strip().split(',')
                image_labels[image_name] = int(label)
        return image_labels

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        image_name = list(self.image_labels.keys())[idx]
        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, image_name)
        label = self.image_labels[image_name]

        image = Image.open(image_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        combined = torch.cat((image, mask), dim=0)

        return combined, label

