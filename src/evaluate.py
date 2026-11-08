import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

THRESHOLDS = {
    "mae": 5.0,
    "rmse": 8.0,
    "r2": 0.70,
}


def main():
    result_path = ROOT_DIR / "training_result.json"

    if not result_path.exists():
        raise FileNotFoundError("training_result.json tidak ditemukan.")

    with open(result_path) as file:
        result = json.load(file)

    print("=== MODEL EVALUATION ===")
    print("Run ID:", result["run_id"])
    print("Parameter:", {
        "n_estimators": result["n_estimators"],
        "max_depth": result["max_depth"],
    })

    passed = True

    for metric, threshold in THRESHOLDS.items():
        value = result[metric]

        if metric == "r2":
            valid = value >= threshold
        else:
            valid = value <= threshold

        print(
            f"{metric.upper()}: {value:.4f} | "
            f"Threshold: {threshold} | "
            f"{'PASSED' if valid else 'FAILED'}"
        )

        passed = passed and valid

    result["validation_status"] = (
        "PASSED" if passed else "FAILED"
    )

    with open(ROOT_DIR / "evaluation_result.json", "w") as file:
        json.dump(result, file, indent=4)

    print("\nValidation:", result["validation_status"])

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()