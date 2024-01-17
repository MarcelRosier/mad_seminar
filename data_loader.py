import os
from typing import List, Tuple

import pandas as pd
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from monai import transforms as monai_transforms
import numpy as np

MEAN = 0
STD = 1


class TrainDataset(Dataset):
    def __init__(self, data: List[str], target_size=(128, 128), transforms=None):
        """
        Loads images from data

        @param data:
            paths to images
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        """
        super(TrainDataset, self).__init__()
        self.target_size = target_size
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.data[idx]).convert("L")
        # Pad to square
        img = transforms.Pad(((img.height - img.width) // 2, 0), fill=0)(img)
        # Resize
        img = img.resize(self.target_size, Image.BICUBIC)
        # Convert to tensor
        img = transforms.ToTensor()(img)

        if self.transforms:
            img = self.transforms(img)

        return img


class TrainDataModule(pl.LightningDataModule):
    def __init__(self, split_dir: str, target_size=(128, 128), batch_size: int = 32):
        """
        Data module for training

        @param split_dir: str
            path to directory containing the split files
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        @param: batch_size: int, default: 32
            batch size
        """
        super(TrainDataModule, self).__init__()
        self.target_size = target_size
        self.batch_size = batch_size

        train_csv_ixi = os.path.join(split_dir, "ixi_normal_train.csv")
        train_csv_fastMRI = os.path.join(split_dir, "normal_train.csv")
        val_csv = os.path.join(split_dir, "normal_val.csv")

        # Load csv files
        train_files_ixi = pd.read_csv(train_csv_ixi)["filename"].tolist()
        train_files_fastMRI = pd.read_csv(train_csv_fastMRI)["filename"].tolist()
        val_files = pd.read_csv(val_csv)["filename"].tolist()

        # Combine files
        self.train_data = train_files_ixi + train_files_fastMRI
        self.val_data = val_files

        # print(f"{self.train_data}")
        # print(f"{self.val_data}")

        # Logging
        print(
            f"Using {len(train_files_ixi)} IXI images "
            f"and {len(train_files_fastMRI)} fastMRI images for training. "
            f"Using {len(val_files)} images for validation."
        )

        self.transforms = transforms.Compose(
            [
                transforms.Normalize((MEAN,), (STD,)),
                transforms.RandomHorizontalFlip(0.1),
                monai_transforms.RandAffine(
                    prob=0.1,
                    spatial_size=target_size,
                    translate_range=(4, 4),
                    rotate_range=(np.pi / 36, np.pi / 36),
                    scale_range=(0.1, 0.1),
                    padding_mode="border",
                ),
            ]
        )

    def train_dataloader(self):
        return DataLoader(
            TrainDataset(self.train_data, self.target_size, transforms=self.transforms),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            TrainDataset(self.val_data, self.target_size, transforms=self.transforms),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )


class TestDataset(Dataset):
    def __init__(
        self,
        img_csv: str,
        pos_mask_csv: str,
        neg_mask_csv: str,
        target_size=(128, 128),
        transforms=None,
    ):
        """
        Loads anomalous images, their positive masks and negative masks from data_dir

        @param img_csv: str
            path to csv file containing filenames to the images
        @param img_csv: str
            path to csv file containing filenames to the positive masks
        @param img_csv: str
            path to csv file containing filenames to the negative masks
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        """
        super(TestDataset, self).__init__()
        self.target_size = target_size
        self.img_paths = pd.read_csv(img_csv)["filename"].tolist()
        self.pos_mask_paths = pd.read_csv(pos_mask_csv)["filename"].tolist()
        self.neg_mask_paths = pd.read_csv(neg_mask_csv)["filename"].tolist()

        self.transforms = transforms

        print(f"{img_csv=} \n {len(self.img_paths)=}")
        assert (
            len(self.img_paths) == len(self.pos_mask_paths) == len(self.neg_mask_paths)
        )
        # print(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.img_paths[idx]).convert("L")
        img = img.resize(self.target_size, Image.BICUBIC)
        img = transforms.ToTensor()(img)
        # Load positive mask
        pos_mask = Image.open(self.pos_mask_paths[idx]).convert("L")
        pos_mask = pos_mask.resize(self.target_size, Image.NEAREST)
        pos_mask = transforms.ToTensor()(pos_mask)

        # Load negative mask
        neg_mask = Image.open(self.neg_mask_paths[idx]).convert("L")
        neg_mask = neg_mask.resize(self.target_size, Image.NEAREST)
        neg_mask = transforms.ToTensor()(neg_mask)

        if self.transforms:
            img = self.transforms(img)
            # pos and neg also?

        return img, pos_mask, neg_mask


class NormalTestDataset(Dataset):
    def __init__(self, img_csv: str, target_size=(128, 128), transforms=None):
        """
        Loads anomalous images, their positive masks and negative masks from data_dir

        @param img_csv: str
            path to csv file containing filenames to the images
        @param img_csv: str
            path to csv file containing filenames to the positive masks
        @param img_csv: str
            path to csv file containing filenames to the negative masks
        @param: target_size: tuple (int, int), default: (128, 128)
            the desired output size
        """
        super(NormalTestDataset, self).__init__()
        self.target_size = target_size
        self.img_paths = pd.read_csv(img_csv)["filename"].tolist()
        self.transforms = transforms
        print(f"{img_csv=} \n {len(self.img_paths)=}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Load image
        img = Image.open(self.img_paths[idx]).convert("L")
        img = img.resize(self.target_size, Image.BICUBIC)
        img = transforms.ToTensor()(img)

        if self.transforms:
            img = self.transforms(img)

        return img


def get_test_dataloader(
    split_dir: str, pathology: str, target_size: Tuple[int, int], batch_size: int
):
    """
    Loads test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param pathology: str
        pathology to load
    @param batch_size: int
        batch size
    """
    img_csv = os.path.join(split_dir, f"{pathology}.csv")
    pos_mask_csv = os.path.join(split_dir, f"{pathology}_ann.csv")
    neg_mask_csv = os.path.join(split_dir, f"{pathology}_neg.csv")

    return DataLoader(
        TestDataset(
            img_csv,
            pos_mask_csv,
            neg_mask_csv,
            target_size,
            transforms=transforms.Compose([transforms.Normalize((MEAN,), (STD,))]),
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def get_normal_test_dataloader(
    split_dir: str, target_size: Tuple[int, int], batch_size: int
):
    """
    Loads test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param pathology: str
        pathology to load
    @param batch_size: int
        batch size
    """
    img_csv = os.path.join(split_dir, f"normal_test.csv")

    return DataLoader(
        NormalTestDataset(
            img_csv,
            target_size,
            transforms=transforms.Compose([transforms.Normalize((MEAN,), (STD,))]),
        ),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )


def get_train_dataloader(split_dir: str, target_size: Tuple[int, int], batch_size: int):
    train_csv_ixi = os.path.join(split_dir, "ixi_normal_train.csv")
    train_csv_fastMRI = os.path.join(split_dir, "normal_train.csv")

    # Load csv files
    train_files_ixi = pd.read_csv(train_csv_ixi)["filename"].tolist()
    train_files_fastMRI = pd.read_csv(train_csv_fastMRI)["filename"].tolist()

    # Combine files
    train_data = train_files_ixi + train_files_fastMRI
    train_transforms = transforms.Compose(
        [
            transforms.Normalize((MEAN,), (STD,)),
            # transforms.RandomHorizontalFlip(),
            monai_transforms.RandAffine(
                prob=0.5,
                spatial_size=target_size,
                translate_range=(4, 4),
                rotate_range=(np.pi / 36, np.pi / 36),
                scale_range=(0.1, 0.1),
                padding_mode="border",
            ),
        ]
    )
    return DataLoader(
        TrainDataset(
            train_data,
            target_size,
            transforms=train_transforms,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )


def get_all_test_dataloaders(
    split_dir: str, target_size: Tuple[int, int], batch_size: int
):
    """
    Loads all test data from split_dir

    @param split_dir: str
        path to directory containing the split files
    @param batch_size: int
        batch size
    """
    pathologies = [
        "absent_septum",
        "artefacts",
        "craniatomy",
        "dural",
        "ea_mass",
        "edema",
        "encephalomalacia",
        "enlarged_ventricles",
        "intraventricular",
        "lesions",
        "mass",
        "posttreatment",
        "resection",
        "sinus",
        "wml",
        "other",
        # "normal_test",
    ]
    return {
        pathology: get_test_dataloader(split_dir, pathology, target_size, batch_size)
        for pathology in pathologies
    }
