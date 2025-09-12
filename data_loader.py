import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from typing import Dict, Any

import torchvision.transforms.functional as TF
import random
from PIL import Image, ImageFilter
import io

class RandomJPEGCompression:
    """Randomly apply JPEG compression artifacts."""
    def __init__(self, quality_range=(30, 90)):
        self.quality_range = quality_range

    def __call__(self, img):
        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class RandomOcclusion:
    """Randomly add a black rectangle (occlusion) to the image."""
    def __init__(self, size_range=(0.1, 0.3), p=0.3):
        self.size_range = size_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        w, h = img.size
        occ_w = int(random.uniform(*self.size_range) * w)
        occ_h = int(random.uniform(*self.size_range) * h)
        x1 = random.randint(0, w - occ_w)
        y1 = random.randint(0, h - occ_h)
        img = img.copy()
        img.paste((0, 0, 0), (x1, y1, x1 + occ_w, y1 + occ_h))
        return img

def build_transforms(train=True, crop_size=(112, 112), grayscale=False, aug_type="standard"):
    t = []
    if grayscale:
        t.append(transforms.Grayscale(num_output_channels=3))
    t.append(transforms.Resize(crop_size))
    if train:
        if aug_type == "standard":
            t += [
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=(-10, 10), translate=(0.05, 0.05), scale=(0.95, 1.05)),
            ]
        elif aug_type == "strong":
            t += [
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.25, hue=0.02),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=(-20, 20), translate=(0.12, 0.12), scale=(0.85, 1.15)),
                transforms.RandomGrayscale(p=0.20),
                transforms.Lambda(lambda img: img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.2))) if random.random() < 0.4 else img),
                RandomJPEGCompression(quality_range=(30, 80)),
                RandomOcclusion(size_range=(0.12, 0.24), p=0.28),
                # Optional: Add random sharpness or brightness adjustments
                transforms.Lambda(lambda img: TF.adjust_sharpness(img, sharpness_factor=random.uniform(0.8, 1.8)) if random.random() < 0.4 else img),
            ]
    else:
        t.append(transforms.CenterCrop(crop_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(
        mean=[0.31928780674934387, 0.2873991131782532, 0.25779902935028076],
        std=[0.19799138605594635, 0.20757903158664703, 0.21088403463363647]
    ))
    return transforms.Compose(t)


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True, test_size=0.1, random_state=42, grayscale=False):
        self.root_dir = root_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []
        self.label_map = {}
        self.train = train

        # Sorted list of class names (students)
        self.classes = sorted([c for c in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, c))])
        for idx, student in enumerate(self.classes):
            self.label_map[idx] = student
            student_dir = os.path.join(root_dir, student)
            for img_file in os.listdir(student_dir):
                self.img_paths.append(os.path.join(student_dir, img_file))
                self.labels.append(idx)

        # Split into train/test
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            self.img_paths, self.labels, test_size=test_size, stratify=self.labels, random_state=random_state
        )
        if self.train:
            self.img_paths = train_paths
            self.labels = train_labels
        else:
            self.img_paths = test_paths
            self.labels = test_labels

        # For snapshot_path meta
        self.meta = {
            "num_classes": len(self.classes),
            "num_images": len(self.img_paths),
            "class_names": self.classes,
            "grayscale": grayscale,
            "transform": repr(transform),
            "test_size": test_size,
        }

    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')  # Ensure 3 channels
        if self.meta["grayscale"]:
            img = img.convert('L').convert('RGB')  # Still outputs 3 channels

        if self.transform:
            img = self.transform(img)
        return img, label

def get_data_loader(root_dir, train=True, crop_size=(112, 112), test_size=0.1, batch_size=32, shuffle=True,
                   num_workers=4, grayscale=False, aug_type="standard"):
    """
    Returns: data_loader, snapshot_meta_dict
    """
    transform = build_transforms(
        train=train, crop_size=crop_size,
        grayscale=grayscale, aug_type=aug_type
    )
    dataset = FaceDataset(
        root_dir=root_dir, transform=transform, train=train,
        test_size=test_size, grayscale=grayscale
    )
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # Collect snapshot meta
    snapshot_meta = dict(
        num_classes=dataset.meta["num_classes"],
        class_names=dataset.meta["class_names"],
        grayscale=grayscale,
        crop_size=crop_size,
        test_size=test_size,
        aug_type=aug_type,
        transform=repr(transform),
        batch_size=batch_size
    )
    return data_loader, snapshot_meta

if __name__ == "__main__":
    root_dir = r"./data"
    grayscale = False
    aug_type = "standard"  # 'standard', 'strong', or 'none'
    crop_size = (112, 112)
    # # Step 1: Compute mean/std once (no augmentation)
    # mean, std = get_mean_std(root_dir, grayscale=grayscale, crop_size=crop_size, batch_size=32)
    # print(f"Computed mean: {mean}, std: {std}")

    # Step 2: Get train loader and meta
    train_loader, train_meta = get_data_loader(
        root_dir=root_dir, train=True, crop_size=crop_size,
        batch_size=4, shuffle=True, grayscale=grayscale, aug_type=aug_type
    )
    print(f"Train snapshot meta: {train_meta}")

    # Step 4: Debug batch and show
    imgs, labels = next(iter(train_loader))
    print(f"Batch: {imgs.shape}, Labels: {labels}")
    flag = "train" if train_loader.dataset.train else "test"

    fig, ax = plt.subplots(1, imgs.shape[0], figsize=(12, 3))
    for i in range(imgs.shape[0]):
        img_np = imgs[i].permute(1, 2, 0).numpy()
        img_np = img_np.clip(0, 1)
        ax[i].imshow(img_np)
        ax[i].set_title(f"Label: {labels[i]}")
        ax[i].axis("off")
    plt.savefig(f"standard_aug_faces_{flag}.png")

