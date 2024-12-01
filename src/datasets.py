import os
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class UnlabeledImageDataset(Dataset):
    def __init__(self, image_list_dir, crop_list, transform=None):
        self.image_list_dir = image_list_dir
        self.transform = transform
        self.crop_list = crop_list

    def __len__(self):
        return len(self.image_list_dir)

    def __getitem__(self, idx):
        img_path = self.image_list_dir[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.crop(self.crop_list[idx])
            if self.transform:
                image = self.transform(image)
            label = -1
            return image, label
        except:
            return None, None

def collate_fn(batch):
    # Filter out None entries
    batch = [item for item in batch if item[0] is not None]
    return torch.utils.data.default_collate(batch)

class UnlabelDataModule(pl.LightningDataModule):
    def __init__(self, image_list_dir, crop_list, batch_size=32, num_workers=4, transform=None):
        super().__init__()
        self.image_list_dir = image_list_dir
        self.crop_list = crop_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
    def setup(self, stage=None):
        self.dataset = UnlabeledImageDataset(self.image_list_dir, self.crop_list, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            collate_fn=collate_fn  # Use custom collate_fn to handle None entries
        )
    def valid_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_fn
        )

class LabeledImageDataset(Dataset):
    def __init__(self, image_list, labels, transform=None):
        self.image_list = image_list
        self.labels = labels  # Add labels as an attribute
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = self.image_list[idx]
        label = self.labels[idx]  # Get the corresponding label
        if self.transform:
            image = self.transform(image)
        return image, label  # Return both image and label


class LabeledDataModule(pl.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )