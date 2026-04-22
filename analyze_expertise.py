from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare prediction CSV files and discover architecture expertise regions."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="Format: name=/path/to/predictions.csv",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bin_size", type=int, default=10)
    parser.add_argument("--top_k_errors", type=int, default=25)
    return parser.parse_args()



def parse_named_inputs(raw_inputs: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for item in raw_inputs:
        if "=" not in item:
            raise ValueError(f"Expected name=path format, got: {item}")
        name, path = item.split("=", 1)
        parsed[name] = Path(path)
    return parsed



def ensure_same_order(frames: dict[str, pd.DataFrame]) -> None:
    reference_name = next(iter(frames))
    reference_paths = list(frames[reference_name]["path"].astype(str))
    for name, frame in frames.items():
        candidate_paths = list(frame["path"].astype(str))
        if candidate_paths != reference_paths:
            raise ValueError(
                "Prediction CSV files are not aligned on the same test sample order. "
                f"Mismatch found in '{name}'."
            )



def add_age_bins(frame: pd.DataFrame, bin_size: int) -> pd.DataFrame:
    frame = frame.copy()
    max_age = int(frame["true_age"].max())
    upper = ((max_age // bin_size) + 1) * bin_size
    bins = np.arange(0, upper + bin_size, bin_size)
    labels = [f"{start}-{start + bin_size - 1}" for start in bins[:-1]]
    frame["age_bin"] = pd.cut(
        frame["true_age"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return frame



def compute_overall_metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame["pred_age"] - frame["true_age"]
    abs_error = error.abs()
    return {
        "mae": float(abs_error.mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "bias": float(error.mean()),
        "within_2": float((abs_error <= 2).mean()),
        "within_5": float((abs_error <= 5).mean()),
        "within_10": float((abs_error <= 10).mean()),
    }



def compute_age_bin_metrics(frames: dict[str, pd.DataFrame], bin_size: int) -> pd.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    for model_name, frame in frames.items():
        enriched = add_age_bins(frame, bin_size=bin_size)
        grouped = enriched.groupby("age_bin", observed=True)
        for age_bin, group in grouped:
            error = group["pred_age"] - group["true_age"]
            abs_error = error.abs()
            rows.append(
                {
                    "model": model_name,
                    "age_bin": str(age_bin),
                    "num_samples": int(len(group)),
                    "mae": float(abs_error.mean()),
                    "rmse": float(np.sqrt((error**2).mean())),
                    "bias": float(error.mean()),
                }
            )
    return pd.DataFrame(rows)



def compute_cohort_metrics(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cohort_bins = [-0.1, 13, 20, 40, 60, 200]
    cohort_labels = ["child", "teen", "young_adult", "adult", "senior"]
    rows: list[dict[str, float | str | int]] = []
    for model_name, frame in frames.items():
        temp = frame.copy()
        temp["cohort"] = pd.cut(temp["true_age"], bins=cohort_bins, labels=cohort_labels)
        for cohort, group in temp.groupby("cohort", observed=True):
            error = group["pred_age"] - group["true_age"]
            abs_error = error.abs()
            rows.append(
                {
                    "model": model_name,
                    "cohort": str(cohort),
                    "num_samples": int(len(group)),
                    "mae": float(abs_error.mean()),
                    "rmse": float(np.sqrt((error**2).mean())),
                    "bias": float(error.mean()),
                }
            )
    return pd.DataFrame(rows)



def compute_expertise_regions(age_bin_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for age_bin, group in age_bin_metrics.groupby("age_bin", observed=True):
        ordered = group.sort_values("mae", ascending=True).reset_index(drop=True)
        if len(ordered) == 0:
            continue
        best = ordered.iloc[0]
        second_mae = float(ordered.iloc[1]["mae"]) if len(ordered) > 1 else float("nan")
        rows.append(
            {
                "age_bin": str(age_bin),
                "best_model": str(best["model"]),
                "best_mae": float(best["mae"]),
                "second_best_mae": second_mae,
                "margin_to_second": float(second_mae - float(best["mae"])) if len(ordered) > 1 else float("nan"),
                "ranking": " > ".join(ordered["model"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)



def compute_sample_level_winners(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_names = list(frames.keys())
    base = frames[model_names[0]][["path", "true_age"]].copy()
    for model_name, frame in frames.items():
        base[f"{model_name}_abs_error"] = frame["abs_error"].values
        base[f"{model_name}_pred"] = frame["pred_age"].values

    error_columns = [f"{model_name}_abs_error" for model_name in model_names]
    winner_indices = np.argmin(base[error_columns].values, axis=1)
    base["best_model_per_sample"] = [model_names[idx] for idx in winner_indices]
    return base



def save_hardest_examples(frames: dict[str, pd.DataFrame], output_dir: Path, top_k: int) -> None:
    hardest_dir = output_dir / "hardest_examples"
    hardest_dir.mkdir(parents=True, exist_ok=True)
    for model_name, frame in frames.items():
        hardest = frame.sort_values("abs_error", ascending=False).head(top_k)
        hardest.to_csv(hardest_dir / f"{model_name}_top_{top_k}.csv", index=False)



def make_plots(overall: pd.DataFrame, age_bin_metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(overall["model"], overall["mae"])
    plt.ylabel("MAE")
    plt.title("Overall MAE by model")
    plt.tight_layout()
    plt.savefig(output_dir / "overall_mae.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name, group in age_bin_metrics.groupby("model", observed=True):
        plt.plot(group["age_bin"], group["mae"], marker="o", label=model_name)
    plt.ylabel("MAE")
    plt.xlabel("True age bin")
    plt.title("Expertise regions: MAE by age bin")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "age_bin_mae.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 6))
    for model_name, group in age_bin_metrics.groupby("model", observed=True):
        plt.plot(group["age_bin"], group["bias"], marker="o", label=model_name)
    plt.axhline(0.0, linestyle="--")
    plt.ylabel("Bias (pred - true)")
    plt.xlabel("True age bin")
    plt.title("Bias by age bin")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "age_bin_bias.png", dpi=200)
    plt.close()



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    named_inputs = parse_named_inputs(args.inputs)
    frames = {name: pd.read_csv(path) for name, path in named_inputs.items()}
    ensure_same_order(frames)

    overall_rows = []
    for model_name, frame in frames.items():
        row = {"model": model_name, **compute_overall_metrics(frame)}
        overall_rows.append(row)
    overall_df = pd.DataFrame(overall_rows).sort_values("mae", ascending=True)

    age_bin_metrics = compute_age_bin_metrics(frames, bin_size=args.bin_size)
    expertise_regions = compute_expertise_regions(age_bin_metrics)
    cohort_metrics = compute_cohort_metrics(frames)
    sample_level_winners = compute_sample_level_winners(frames)

    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)
    age_bin_metrics.to_csv(output_dir / "age_bin_metrics.csv", index=False)
    expertise_regions.to_csv(output_dir / "expertise_regions.csv", index=False)
    cohort_metrics.to_csv(output_dir / "cohort_metrics.csv", index=False)
    sample_level_winners.to_csv(output_dir / "sample_level_winners.csv", index=False)
    save_hardest_examples(frames, output_dir, top_k=args.top_k_errors)
    make_plots(overall_df, age_bin_metrics, output_dir)

    summary = {
        "best_overall_model": overall_df.iloc[0]["model"],
        "best_overall_mae": float(overall_df.iloc[0]["mae"]),
        "num_models": int(len(overall_df)),
        "num_test_samples": int(len(next(iter(frames.values())))),
        "num_expertise_bins": int(len(expertise_regions)),
    }
    with (output_dir / "analysis_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("Overall metrics")
    print(overall_df.to_string(index=False))
    print("\nExpertise regions")
    print(expertise_regions.to_string(index=False))
    print(f"\nSaved analysis to: {output_dir}")


if __name__ == "__main__":
    main()
