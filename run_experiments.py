from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch

from models import build_model, count_trainable_parameters, format_parameter_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standardized experiments for the proposal.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--python_executable", type=str, default=sys.executable)
    parser.add_argument("--mode", type=str, choices=["architectures", "losses", "schedules", "sampling"], default="architectures")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--pool_splits", type=str, nargs="+", default=["part1", "part2"])
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--channels", type=int, nargs="+", default=[45, 90, 180, 270, 360])
    parser.add_argument("--blocks_per_stage", type=int, nargs="+", default=[2, 2, 3, 3, 3])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unet_decoder_scale", type=float, default=0.12)
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--loss", type=str, choices=["rmse", "mse", "mae", "huber"], default="rmse")
    parser.add_argument("--sampling", type=str, choices=["uniform", "balanced_classes"], default="balanced_classes")
    parser.add_argument("--resnet_pretrained", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("CUDA requested, but this PyTorch build does not have CUDA enabled. Falling back to CPU.")
        return "cpu"
    return device_arg


def run_command(command: list[str]) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def print_parameter_report(args: argparse.Namespace) -> None:
    print("Parameter count report")
    for model_name in ["vgg", "resvgg", "unet", "resnet50"]:
        model = build_model(
            model_name,
            channels=args.channels,
            blocks_per_stage=args.blocks_per_stage,
            dropout=args.dropout,
            unet_decoder_scale=args.unet_decoder_scale,
            resnet_pretrained=args.resnet_pretrained,
        )
        params = count_trainable_parameters(model)
        print(f"  {model_name:<8} -> {format_parameter_count(params)} ({params:,})")


def build_common_train_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--data_root",
        args.data_root,
        "--output_dir",
        str(Path(args.output_root) / "train_runs"),
        "--pool_splits",
        *args.pool_splits,
        "--val_ratio",
        str(args.val_ratio),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--image_size",
        str(args.image_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--channels",
        *[str(x) for x in args.channels],
        "--blocks_per_stage",
        *[str(x) for x in args.blocks_per_stage],
        "--dropout",
        str(args.dropout),
        "--unet_decoder_scale",
        str(args.unet_decoder_scale),
        "--optimizer",
        args.optimizer,
        "--lr",
        str(args.lr),
        "--loss",
        args.loss,
        "--sampling",
        args.sampling,
    ]
    if args.resnet_pretrained:
        common.append("--resnet_pretrained")
    if args.amp and args.device == "cuda":
        common.append("--amp")
    return common


def train_and_evaluate(args: argparse.Namespace, experiment_name: str, extra_train_args: list[str]) -> Path:
    output_root = Path(args.output_root)
    train_output_root = output_root / "train_runs"
    eval_output_root = output_root / "eval_runs"
    train_output_root.mkdir(parents=True, exist_ok=True)
    eval_output_root.mkdir(parents=True, exist_ok=True)

    train_command = [args.python_executable, "train.py", *build_common_train_args(args), "--experiment_name", experiment_name, *extra_train_args]
    run_command(train_command)

    checkpoint_path = train_output_root / experiment_name / "best.pt"
    eval_command = [
        args.python_executable,
        "evaluate.py",
        "--checkpoint",
        str(checkpoint_path),
        "--data_root",
        args.data_root,
        "--output_dir",
        str(eval_output_root / experiment_name),
        "--batch_size",
        str(args.batch_size),
        "--image_size",
        str(args.image_size),
        "--num_workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    run_command(eval_command)
    return eval_output_root / experiment_name / "predictions.csv"


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    print(f"Resolved device: {args.device}")
    print_parameter_report(args)

    collected_predictions: list[str] = []

    if args.mode == "architectures":
        experiments = [
            ("arch_vgg", ["--model", "vgg"]),
            ("arch_resvgg", ["--model", "resvgg"]),
            ("arch_unet", ["--model", "unet"]),
            ("sanity_resnet50", ["--model", "resnet50"]),
        ]
    elif args.mode == "losses":
        experiments = [
            ("loss_rmse", ["--model", "resvgg", "--loss", "rmse"]),
            ("loss_huber", ["--model", "resvgg", "--loss", "huber"]),
            ("loss_mae", ["--model", "resvgg", "--loss", "mae"]),
        ]
    elif args.mode == "schedules":
        experiments = [
            ("sched_cosine", ["--model", "resvgg", "--scheduler", "cosine"]),
            ("sched_plateau", ["--model", "resvgg", "--scheduler", "plateau"]),
            ("sched_onecycle", ["--model", "resvgg", "--scheduler", "onecycle"]),
        ]
    else:
        experiments = [
            ("sampling_uniform", ["--model", "resvgg", "--sampling", "uniform"]),
            ("sampling_balanced_classes", ["--model", "resvgg", "--sampling", "balanced_classes"]),
        ]

    for experiment_name, extra_train_args in experiments:
        prediction_csv = train_and_evaluate(args, experiment_name, extra_train_args)
        collected_predictions.append(f"{experiment_name}={prediction_csv}")

    analysis_dir = Path(args.output_root) / f"analysis_{args.mode}"
    analyze_command = [
        args.python_executable,
        "analyze_expertise.py",
        "--inputs",
        *collected_predictions,
        "--output_dir",
        str(analysis_dir),
    ]
    run_command(analyze_command)


if __name__ == "__main__":
    main()
