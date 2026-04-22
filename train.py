from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    AgeEstimationDataset,
    build_split_audit,
    build_transforms,
    make_exact_age_weighted_sampler,
    scan_multiple_splits,
    split_train_val_by_age,
    summarize_age_bin_counts,
    summarize_exact_age_counts,
    write_label_count_csv,
)
from losses import build_loss
from models import build_model, count_trainable_parameters, format_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train age estimation models for the proposal study.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--pool_splits", type=str, nargs="+", default=["part1", "part2"])
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_split", type=str, default="part3")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default=None)

    parser.add_argument("--model", type=str, choices=["vgg", "resvgg", "unet", "resnet50"], required=True)
    parser.add_argument("--channels", type=int, nargs="+", default=[45, 90, 180, 270, 360])
    parser.add_argument("--blocks_per_stage", type=int, nargs="+", default=[2, 2, 3, 3, 3])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unet_decoder_scale", type=float, default=0.12)
    parser.add_argument("--resnet_pretrained", action="store_true")

    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--loss", type=str, choices=["rmse", "mse", "mae", "huber"], default="rmse")
    parser.add_argument("--huber_delta", type=float, default=5.0)
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "plateau", "onecycle"], default="cosine")
    parser.add_argument("--sampling", type=str, choices=["uniform", "balanced_classes"], default="balanced_classes")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested, but this PyTorch build does not have CUDA enabled. Falling back to CPU.")
        return "cpu"
    return device_arg


def compute_regression_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    errors = predictions - targets
    abs_errors = np.abs(errors)
    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt((errors**2).mean())),
        "bias": float(errors.mean()),
        "within_5": float((abs_errors <= 5).mean()),
        "within_10": float((abs_errors <= 10).mean()),
    }


def build_optimizer(args: argparse.Namespace, model: torch.nn.Module):
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def build_scheduler(args: argparse.Namespace, optimizer, steps_per_epoch: int):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.scheduler == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    if args.scheduler == "onecycle":
        total_steps = max(1, args.epochs * steps_per_epoch)
        return OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def evaluate_one_epoch(model, loader, criterion, device: torch.device, use_amp: bool):
    model.eval()
    total_loss = 0.0
    total_items = 0
    predictions: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets, _ in tqdm(loader, desc="validate", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
            predictions.append(outputs.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())

    predictions_np = np.concatenate(predictions)
    targets_np = np.concatenate(targets_list)
    metrics = compute_regression_metrics(predictions_np, targets_np)
    metrics["loss"] = total_loss / max(1, total_items)
    return metrics


def create_train_loader(args: argparse.Namespace, dataset: AgeEstimationDataset, device: torch.device) -> DataLoader:
    sampler = None
    shuffle = args.sampling == "uniform"
    if args.sampling == "balanced_classes":
        sampler = make_exact_age_weighted_sampler(dataset)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(resolve_device(args.device))
    use_amp = args.amp and device.type == "cuda"

    run_name = args.experiment_name or f"{args.model}_{args.loss}_{args.optimizer}_{args.sampling}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    pooled_samples = scan_multiple_splits(args.data_root, args.pool_splits)
    train_samples, val_samples = split_train_val_by_age(
        pooled_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    split_audit = build_split_audit(pooled_samples, train_samples, val_samples)
    if split_audit["missing_ages_in_train"]:
        raise RuntimeError(f"Some ages disappeared from train split: {split_audit['missing_ages_in_train']}")

    train_dataset = AgeEstimationDataset(train_samples, transform=build_transforms(image_size=args.image_size, train=True))
    val_dataset = AgeEstimationDataset(val_samples, transform=build_transforms(image_size=args.image_size, train=False))

    train_loader = create_train_loader(args, train_dataset, device)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model_kwargs = {
        "channels": args.channels,
        "blocks_per_stage": args.blocks_per_stage,
        "dropout": args.dropout,
        "unet_decoder_scale": args.unet_decoder_scale,
        "resnet_pretrained": args.resnet_pretrained,
    }
    model = build_model(args.model, **model_kwargs).to(device)
    param_count = count_trainable_parameters(model)

    criterion = build_loss(args.loss, huber_delta=args.huber_delta)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, steps_per_epoch=len(train_loader))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    config_to_save = vars(args).copy()
    config_to_save["resolved_device"] = str(device)
    config_to_save["parameter_count"] = param_count
    config_to_save["parameter_count_human"] = format_parameter_count(param_count)
    config_to_save["split_audit"] = split_audit
    config_to_save["train_age_counts"] = summarize_exact_age_counts(train_samples)
    config_to_save["val_age_counts"] = summarize_exact_age_counts(val_samples)
    config_to_save["train_age_bin_counts"] = summarize_age_bin_counts(train_samples, bin_size=10)
    config_to_save["val_age_bin_counts"] = summarize_age_bin_counts(val_samples, bin_size=10)
    save_json(config_to_save, output_dir / "config.json")
    write_label_count_csv(output_dir / "label_counts.csv", pooled_samples, train_samples, val_samples)

    history_path = output_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_mae",
                "val_rmse",
                "val_bias",
                "val_within_5",
                "val_within_10",
                "lr",
            ]
        )

    print(f"Run name: {run_name}")
    print(f"Pooled splits: {args.pool_splits}")
    print(
        f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
        f"Actual val fraction: {split_audit['actual_val_fraction']:.4f}"
    )
    print(
        f"Distinct ages -> pooled: {split_audit['num_distinct_ages_pooled']}, "
        f"train: {split_audit['num_distinct_ages_train']}, val: {split_audit['num_distinct_ages_val']}"
    )
    print(f"Missing ages in train: {split_audit['missing_ages_in_train']}")
    print(f"Model: {args.model} | Parameters: {format_parameter_count(param_count)} ({param_count:,})")
    print(f"Loss: {args.loss} | Optimizer: {args.optimizer} | LR: {args.lr} | Scheduler: {args.scheduler}")
    print(f"Sampling: {args.sampling} | Mini-batch size: {args.batch_size}")

    best_val_rmse = math.inf
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0

        progress = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False)
        for images, targets, _ in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()

            batch_size = images.size(0)
            running_loss += float(loss.item()) * batch_size
            running_items += batch_size
            progress.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(1, running_items)
        val_metrics = evaluate_one_epoch(model, val_loader, criterion, device, use_amp)

        if args.scheduler == "plateau" and scheduler is not None:
            scheduler.step(val_metrics["rmse"])
        elif args.scheduler == "cosine" and scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        with history_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.8f}",
                    f"{val_metrics['loss']:.8f}",
                    f"{val_metrics['mae']:.8f}",
                    f"{val_metrics['rmse']:.8f}",
                    f"{val_metrics['bias']:.8f}",
                    f"{val_metrics['within_5']:.8f}",
                    f"{val_metrics['within_10']:.8f}",
                    f"{current_lr:.10f}",
                ]
            )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_rmse={val_metrics['rmse']:.4f} | val_mae={val_metrics['mae']:.4f} | lr={current_lr:.6g}"
        )

        improved = val_metrics["rmse"] < best_val_rmse
        if improved:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            bad_epochs = 0
            checkpoint = {
                "epoch": epoch,
                "best_val_rmse": best_val_rmse,
                "model_name": args.model,
                "model_kwargs": model_kwargs,
                "state_dict": model.state_dict(),
                "train_args": vars(args),
                "parameter_count": param_count,
                "split_audit": split_audit,
            }
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= args.patience:
            print(f"Early stopping triggered after {args.patience} non-improving epochs.")
            break

    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "parameter_count": param_count,
            "parameter_count_human": format_parameter_count(param_count),
            "split_audit": split_audit,
        },
        output_dir / "training_summary.json",
    )
    print(f"Finished. Best epoch={best_epoch}, best_val_rmse={best_val_rmse:.4f}")
    print(f"Checkpoint saved to: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
