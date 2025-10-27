import os
import zipfile
import logging
import urllib.request
from typing import Optional

import cv2
import numpy as np
import albumentations as A

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VOCSegmentation

from .corruption import get_corruption_transforms

VOC_COLORMAP = [
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128],
]


class PascalVOCDataset(VOCSegmentation):
    def __init__(
        self,
        root: str = "./data",
        year: str = "2012",
        image_set: str = "train",
        download: bool = True,
        transform: Optional[A.Compose] = None,
        use_index_label: bool = True,
    ):
        super().__init__(
            root=root,
            year=year,
            image_set=image_set,
            download=download,
            transform=transform,
        )
        self.n_classes = 21
        self.transform = transform
        self.use_index_label = use_index_label

    @staticmethod
    def _convert_to_segmentation_mask(
        mask: np.ndarray, use_index_label: bool = True
    ) -> np.ndarray:
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros(
            (height, width, len(VOC_COLORMAP)),
            dtype=np.float32,
        )
        for label_index, label in enumerate(VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(
                mask == label, axis=-1
            ).astype(float)

        if use_index_label:
            segmentation_mask = np.argmax(segmentation_mask, axis=-1)
        return segmentation_mask

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(self.images[index])
        mask = cv2.imread(self.masks[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        mask = self._convert_to_segmentation_mask(mask, self.use_index_label)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.moveaxis(image, -1, 0) / 255
        return image, mask


class ADE20kDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "training",
        transform: Optional[A.Compose] = None,
    ):
        self.root = root
        self.split = split
        self.n_classes = 150
        self.transform = transform

        root = os.path.join(root, "ADEChallengeData2016")
        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir = os.path.join(root, "annotations", split)

        # Check if the dataset is already downloaded
        if not os.path.exists(self.images_dir) or not os.path.exists(self.masks_dir):
            self.download_and_extract_dataset()

        self.image_files = os.listdir(self.images_dir)

    def download_and_extract_dataset(self) -> None:
        dataset_url = (
            "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
        )
        zip_path = os.path.join(self.root, "ADEChallengeData2016.zip")
        os.makedirs(self.root, exist_ok=True)

        logging.info("Downloading dataset...")
        urllib.request.urlretrieve(dataset_url, zip_path)

        logging.info("Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.root)

        logging.info("Dataset extracted!")
        os.remove(zip_path)

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        img_name = self.image_files[index]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) - 1
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.moveaxis(image, -1, 0) / 255
        return image, mask


def _add_mod_suffix(stem: str) -> str:
    stem = os.path.splitext(stem)[0]
    return stem if stem.endswith("_0000") else stem + "_0000"


def _strip_mod_suffix(stem: str) -> str:
    return stem[:-5] if stem.endswith("_0000") else stem


class LiverUSDataset(Dataset):
    def __init__(
        self,
        root: str,
        transform: Optional[A.Compose] = None,
        valid_ids: tuple[int, ...] = (0, 1, 2),
        ignore_index: int = 255,
        subset_ids: Optional[list[str]] = None,  # <- nuevo
        folder_name: str = "Dataset771_livervsi",  # <- nuevo
    ):
        # en LiverUSDataset.__init__
        base = os.path.join(
            root, folder_name
        )  # antes: os.path.join(root, "liver", split)
        self.images_dir = os.path.join(base, "imagesTr")
        self.masks_dir = os.path.join(base, "labelsTr")
        self.transform = transform
        self.valid_ids = set(valid_ids)
        self.ignore_index = ignore_index
        self.n_classes = len(valid_ids)
        self.subset_ids = set(subset_ids) if subset_ids is not None else None

        assert os.path.isdir(self.images_dir), f"Not found: {self.images_dir}"
        assert os.path.isdir(self.masks_dir), f"Not found: {self.masks_dir}"

        # dentro de LiverUSDataset.__init__
        files = [
            f
            for f in os.listdir(self.images_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        if self.subset_ids is not None:
            wanted = {
                _add_mod_suffix(x.lower()) for x in self.subset_ids
            }  # <- agrega _0000
            files = [f for f in files if os.path.splitext(f.lower())[0] in wanted]
            if not files:
                raise RuntimeError(
                    "subset_ids dejó el split vacío (revisa nombres y sufijo _0000)."
                )

        self.image_files = sorted(files)

    def __len__(self):
        return len(self.image_files)

    def _remap_invalid_to_ignore(self, mask: np.ndarray) -> np.ndarray:
        invalid = ~np.isin(mask, list(self.valid_ids))
        if invalid.any():
            mask = mask.copy()
            mask[invalid] = self.ignore_index
        return mask

    def __getitem__(self, idx: int):
        # --- dentro de LiverUSDataset.__getitem__ ---
        img_name = self.image_files[idx]
        img_stem = os.path.splitext(img_name)[0]

        # máscaras NO tienen _0000
        mask_stem = _strip_mod_suffix(img_stem)
        msk_path = os.path.join(self.masks_dir, mask_stem + ".png")

        image_bgr = cv2.imread(os.path.join(self.images_dir, img_name))
        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        # comprobaciones claras (evitan AttributeError si falla la lectura)
        if image_bgr is None:
            raise FileNotFoundError(
                f"No pude leer imagen: {os.path.join(self.images_dir, img_name)}"
            )
        if mask is None:
            raise FileNotFoundError(f"No pude leer máscara (sin _0000): {msk_path}")

        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = self._remap_invalid_to_ignore(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = np.moveaxis(image, -1, 0).astype(np.float32) / 255.0
        mask = mask.astype(np.int64)
        return image, mask


# splits_utils.py
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_fold_split(json_path: str | Path, fold: int) -> Dict[str, List[str]]:
    """
    Lee un JSON con una lista de folds: [{ "train":[...], "val":[...] }, ...]
    Devuelve dict con keys: 'train', 'val'. Si 'val' no existe, lo infiere.
    """
    json_path = Path(json_path)
    with json_path.open("r") as f:
        folds = json.load(f)

    if not isinstance(folds, list):
        raise ValueError("El JSON debe ser una lista de objetos (uno por fold).")

    if fold < 0 or fold >= len(folds):
        raise IndexError(f"fold={fold} fuera de rango (0..{len(folds) - 1}).")

    entry = folds[fold]
    if "train" not in entry:
        raise KeyError(f"El fold {fold} no contiene la clave 'train'.")

    train_ids = list(entry["train"])
    val_ids = list(entry.get("val", []))

    # Si no hay 'val', lo inferimos del complemento de train dentro del universo total
    if not val_ids:
        # Universo = unión de todos los ids presentes en todos los folds
        all_ids = set()
        for e in folds:
            all_ids.update(e.get("train", []))
            all_ids.update(e.get("val", []))
        val_ids = sorted(list(all_ids.difference(set(train_ids))))

    return {"train": train_ids, "val": val_ids}


def split_ids_for_fold(json_path: str | Path, fold: int) -> Tuple[List[str], List[str]]:
    s = load_fold_split(json_path, fold)
    return s["train"], s["val"]


def get_dataloader(
    dataset_name: str,
    img_dim=(490, 490),
    batch_size: int = 6,
    corruption_severity: int | None = None,
    root: str = "./data",
    split_json: str | None = None,
    folder_name: str = "Dataset771_livervsi",
    n_workers: int = 32,
    fold: int | None = None,
):
    """Get the dataloaders for Pascal VOC (voc) or ADE20k (ade20k) or liver.

    Args:
        dataset_name (str): The name of the dataset either, `voc` or `ade20k` or `liver`.
        img_dim (tuple[int, int], optional): The input size of the images.
            Defaults to (490, 490).
        batch_size (int, optional): The batch size of the dataloader. Defaults to 6.
        corruption_severity (int, optional): The corruption severity level between 1 and 5.
            Defaults to None.

    Returns:
        tuple[DataLoader, DataLoader]: The train and validation loader respectively.
    """
    assert dataset_name in ["ade20k", "voc", "liver"], (
        "dataset name not in [ade20k, voc, liver]"
    )
    transform = A.Compose([A.Resize(height=img_dim[0], width=img_dim[1])])

    train_ids = val_ids = None
    if dataset_name == "liver" and split_json is not None and fold is not None:
        train_ids, val_ids = split_ids_for_fold(split_json, fold)

    if dataset_name == "liver":
        train_dataset = LiverUSDataset(
            root=root,
            folder_name=folder_name,
            transform=transform,
            subset_ids=train_ids,
        )

        if corruption_severity is not None:
            transform = get_corruption_transforms(img_dim, corruption_severity)
        val_dataset = LiverUSDataset(
            root=root, folder_name=folder_name, transform=transform, subset_ids=val_ids
        )
    if dataset_name == "voc":
        train_dataset = PascalVOCDataset(
            root="./data",
            year="2012",
            image_set="train",
            download=True,
            transform=transform,
        )

        if corruption_severity is not None:
            transform = get_corruption_transforms(img_dim, corruption_severity)
        val_dataset = PascalVOCDataset(
            root="./data",
            year="2012",
            image_set="val",
            download=False,
            transform=transform,
        )
    elif dataset_name == "ade20k":
        train_dataset = ADE20kDataset(
            root="./data",
            split="training",
            transform=transform,
        )

        if corruption_severity is not None:
            transform = get_corruption_transforms(img_dim, corruption_severity)
        val_dataset = ADE20kDataset(
            root="./data",
            split="validation",
            transform=transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=n_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        shuffle=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
