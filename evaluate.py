from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import AgeEstimationDataset, build_transforms
from models import build_model, count_trainable_parameters, format_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained age estimation model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--test_split", type=str, default="part3")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--latency_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested, but this PyTorch build does not have CUDA enabled. Falling back to CPU.")
        return "cpu"
    return device_arg


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    abs_errors = np.abs(errors)
    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "bias": float(errors.mean()),
        "within_2": float((abs_errors <= 2).mean()),
        "within_5": float((abs_errors <= 5).mean()),
        "within_10": float((abs_errors <= 10).mean()),
    }


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(
        checkpoint["model_name"],
        **checkpoint["model_kwargs"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return checkpoint, model


def measure_latency(model, dataset, device: torch.device, max_samples: int, num_workers: int) -> dict[str, float]:
    if max_samples <= 0:
        return {"latency_ms_mean": float("nan"), "latency_ms_std": float("nan"), "latency_num_samples": 0}

    subset_size = min(max_samples, len(dataset))
    subset = Subset(dataset, range(subset_size))
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    timings_ms: list[float] = []
    warmup_batches = 5
    with torch.no_grad():
        for batch_index, (images, _, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed_ms = (end - start) * 1000.0
            if batch_index >= warmup_batches:
                timings_ms.append(elapsed_ms)

    if len(timings_ms) == 0:
        return {"latency_ms_mean": float("nan"), "latency_ms_std": float("nan"), "latency_num_samples": 0}

    return {
        "latency_ms_mean": float(np.mean(timings_ms)),
        "latency_ms_std": float(np.std(timings_ms)),
        "latency_num_samples": int(len(timings_ms)),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(resolve_device(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint, model = load_checkpoint(args.checkpoint, device)
    dataset = AgeEstimationDataset.from_split(
        data_root=args.data_root,
        split=args.test_split,
        transform=build_transforms(image_size=args.image_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    all_rows: list[dict[str, float | str | int]] = []
    predictions_accumulator: list[np.ndarray] = []
    targets_accumulator: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets, paths in tqdm(loader, desc="test"):
            images = images.to(device, non_blocking=True)
            outputs = model(images).detach().cpu().numpy()
            targets_np = targets.numpy()
            predictions_accumulator.append(outputs)
            targets_accumulator.append(targets_np)

            for path, target, prediction in zip(paths, targets_np, outputs):
                abs_error = abs(float(prediction) - float(target))
                all_rows.append(
                    {
                        "path": path,
                        "true_age": int(round(float(target))),
                        "pred_age": float(prediction),
                        "signed_error": float(prediction - target),
                        "abs_error": abs_error,
                        "age_bin_10": f"{int(target // 10) * 10}-{int(target // 10) * 10 + 9}",
                    }
                )

    predictions = np.concatenate(predictions_accumulator)
    targets = np.concatenate(targets_accumulator)
    metrics = compute_metrics(predictions, targets)
    latency = measure_latency(
        model=model,
        dataset=dataset,
        device=device,
        max_samples=args.latency_samples,
        num_workers=args.num_workers,
    )

    predictions_csv = output_dir / "predictions.csv"
    summary_json = output_dir / "summary.json"
    pd.DataFrame(all_rows).to_csv(predictions_csv, index=False)

    summary = {
        "checkpoint": str(args.checkpoint),
        "model_name": checkpoint["model_name"],
        "parameter_count": checkpoint.get("parameter_count", count_trainable_parameters(model)),
        "parameter_count_human": format_parameter_count(
            checkpoint.get("parameter_count", count_trainable_parameters(model))
        ),
        "num_test_samples": len(dataset),
        **metrics,
        **latency,
    }
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Evaluation summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print(f"Predictions CSV: {predictions_csv}")
    print(f"Summary JSON:   {summary_json}")


if __name__ == "__main__":
    main()
