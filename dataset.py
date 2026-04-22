from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)
Sample = Tuple[Path, int]


def parse_age_from_filename(filename: str) -> int:
    stem = Path(filename).stem
    first_token = stem.split("_")[0]
    try:
        return int(first_token)
    except ValueError as exc:
        raise ValueError(f"Could not parse age from file name: {filename}") from exc


def scan_split_directory(
    data_root: str | Path,
    split: str,
    allowed_extensions: Sequence[str] = tuple(IMAGE_EXTENSIONS),
) -> list[Sample]:
    data_root = Path(data_root)
    candidate = data_root / split
    split_dir = candidate if candidate.exists() else Path(split)
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Could not find split folder '{split}'. Checked '{candidate}' and '{Path(split)}'."
        )

    extensions = {ext.lower() for ext in allowed_extensions}
    samples: list[Sample] = []
    for path in sorted(split_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            samples.append((path, parse_age_from_filename(path.name)))

    if len(samples) == 0:
        raise RuntimeError(f"No images were found in {split_dir}")
    return samples


def scan_multiple_splits(
    data_root: str | Path,
    splits: Sequence[str],
    allowed_extensions: Sequence[str] = tuple(IMAGE_EXTENSIONS),
) -> list[Sample]:
    pooled: list[Sample] = []
    for split in splits:
        pooled.extend(scan_split_directory(data_root, split, allowed_extensions=allowed_extensions))
    if len(pooled) == 0:
        raise RuntimeError(f"No images were found in pooled splits: {splits}")
    return pooled


class AgeEstimationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable | None = None,
    ) -> None:
        self.samples: list[Sample] = [(Path(path), int(age)) for path, age in samples]
        self.transform = transform
        if len(self.samples) == 0:
            raise RuntimeError("Dataset is empty.")

    @classmethod
    def from_split(
        cls,
        data_root: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> "AgeEstimationDataset":
        return cls(samples=scan_split_directory(data_root, split), transform=transform)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, age = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(float(age), dtype=torch.float32)
        return image, target, str(path)

    @property
    def ages(self) -> list[int]:
        return [age for _, age in self.samples]


def build_transforms(
    image_size: int = 224,
    train: bool = False,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
):
    resize = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)

    if train:
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))],
                    p=0.15,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            resize,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def split_train_val_by_age(
    samples: Sequence[Sample],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample]]:
    """
    Build a new train/validation split from the pooled part1+part2 set.
    The split is stratified by exact integer age so that every age label stays in train.
    Singleton ages are kept in the training set.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    rng = random.Random(seed)
    age_to_items: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        age_to_items[int(sample[1])].append(sample)

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []

    for age in sorted(age_to_items.keys()):
        items = age_to_items[age][:]
        rng.shuffle(items)
        num_items = len(items)

        if num_items == 1:
            val_count = 0
        else:
            val_count = max(1, int(round(num_items * val_ratio)))
            val_count = min(val_count, num_items - 1)

        val_samples.extend(items[:val_count])
        train_samples.extend(items[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def summarize_exact_age_counts(samples: Sequence[Sample]) -> dict[int, int]:
    return dict(sorted(Counter(int(age) for _, age in samples).items(), key=lambda item: item[0]))


def summarize_age_bin_counts(samples: Sequence[Sample], bin_size: int = 10) -> dict[str, int]:
    counts = Counter((int(age) // bin_size) * bin_size for _, age in samples)
    return {
        f"{start}-{start + bin_size - 1}": int(count)
        for start, count in sorted(counts.items(), key=lambda item: item[0])
    }


def build_split_audit(
    pooled_samples: Sequence[Sample],
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
) -> dict[str, object]:
    pooled_ages = {int(age) for _, age in pooled_samples}
    train_ages = {int(age) for _, age in train_samples}
    val_ages = {int(age) for _, age in val_samples}

    pooled_counts = summarize_exact_age_counts(pooled_samples)
    train_counts = summarize_exact_age_counts(train_samples)
    val_counts = summarize_exact_age_counts(val_samples)

    return {
        "num_pooled_samples": int(len(pooled_samples)),
        "num_train_samples": int(len(train_samples)),
        "num_val_samples": int(len(val_samples)),
        "actual_val_fraction": float(len(val_samples) / max(1, len(pooled_samples))),
        "num_distinct_ages_pooled": int(len(pooled_ages)),
        "num_distinct_ages_train": int(len(train_ages)),
        "num_distinct_ages_val": int(len(val_ages)),
        "missing_ages_in_train": sorted(pooled_ages - train_ages),
        "missing_ages_in_val": sorted(pooled_ages - val_ages),
        "min_train_count_per_age": int(min(train_counts.values())),
        "max_train_count_per_age": int(max(train_counts.values())),
        "min_val_count_per_age": int(min(val_counts.values())) if val_counts else 0,
        "max_val_count_per_age": int(max(val_counts.values())) if val_counts else 0,
    }


def write_label_count_csv(
    output_path: str | Path,
    pooled_samples: Sequence[Sample],
    train_samples: Sequence[Sample],
    val_samples: Sequence[Sample],
) -> None:
    pooled_counts = summarize_exact_age_counts(pooled_samples)
    train_counts = summarize_exact_age_counts(train_samples)
    val_counts = summarize_exact_age_counts(val_samples)
    all_ages = sorted(set(pooled_counts) | set(train_counts) | set(val_counts))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["age", "pooled_count", "train_count", "val_count"])
        for age in all_ages:
            writer.writerow([age, pooled_counts.get(age, 0), train_counts.get(age, 0), val_counts.get(age, 0)])


def make_exact_age_weighted_sampler(
    dataset: AgeEstimationDataset,
    temperature: float = 1.0,
) -> WeightedRandomSampler:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    counts = Counter(int(age) for age in dataset.ages)
    weights = [(1.0 / counts[int(age)]) ** temperature for age in dataset.ages]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
