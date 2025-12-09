import os
import json
import pandas as pd

# --- CONFIGURATION ---
TASKS_FILE = "bert-tasks.xlsx"  # Expected cols: dataset_name, subset
MODELS_FILE = "bert-models.xlsx"  # Expected col: model_name
RESULTS_DIR = "experiment_results"
OUTPUT_CSV = "final_aggregated_results.csv"
OUTPUT_PIVOT = "final_pivot_accuracy.csv"


def aggregate_results():
    print("--- Starting Aggregation ---")

    # 1. Load Reference Files
    try:
        tasks_df = pd.read_excel(TASKS_FILE)
        models_df = pd.read_excel(MODELS_FILE)
    except Exception as e:
        print(f"Error loading Excel files: {e}")
        return

    duplicated_tasks = tasks_df.duplicated(subset=["dataset_name", "subset"]).sum()
    if duplicated_tasks > 0:
        print(f"Warning: Found {duplicated_tasks} duplicated task entries in {TASKS_FILE}")

    results_data = []

    # 2. Iterate through every possible Task + Model combination
    # We use the Excel files as the "Source of Truth"
    total_combinations = len(tasks_df) * len(models_df)
    processed_count = 0

    print(f"Found {len(tasks_df)} tasks and {len(models_df)} models.")
    print(f"Scanning for {total_combinations} potential experiments...")

    for _, task_row in tasks_df.iterrows():
        raw_dataset = str(task_row["dataset_name"]).strip()

        # Handle subset: Check if it's NaN or empty
        raw_subset = task_row["subset"]
        has_subset = pd.notna(raw_subset) and str(raw_subset).strip() != ""

        # Sanitization Rule 1: Task name replace / with _
        safe_dataset = raw_dataset.replace("/", "_")

        if has_subset:
            raw_subset = str(raw_subset).strip()
            safe_subset = raw_subset.replace("/", "_")

        for _, model_row in models_df.iterrows():
            raw_model = str(model_row["model_name"]).strip()

            # Sanitization Rule 2: Model name replace / with -
            safe_model = raw_model.replace("/", "-")

            # 3. Construct Directory Name
            # Logic: {task_name}@{subset}_{model_name} OR {task_name}_{model_name}
            if has_subset:
                folder_name = f"{safe_dataset}@{safe_subset}_{safe_model}"
                display_name = f"{raw_dataset} ({raw_subset})"
            else:
                folder_name = f"{safe_dataset}_{safe_model}"
                display_name = f"{raw_dataset}"

            # 4. Check File Existence
            file_path = os.path.join(RESULTS_DIR, folder_name, "test_results.json")

            record = {
                "dataset_name": raw_dataset,
                "subset": raw_subset if has_subset else None,
                "model_name": raw_model,
                "folder_name": folder_name,
                "status": "Missing",
            }

            if os.path.exists(file_path):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Extract keys based on your provided JSON structure
                    record.update(
                        {
                            "status": "Completed",
                            "eval_accuracy": data.get("eval_accuracy"),
                            "eval_f1": data.get("eval_f1"),
                            "eval_precision": data.get("eval_precision"),
                            "eval_recall": data.get("eval_recall"),
                            "eval_loss": data.get("eval_loss"),
                            "epoch": data.get("epoch"),
                            "runtime": data.get("eval_runtime"),
                        }
                    )
                except Exception as e:
                    print(f"Error reading JSON for {folder_name}: {e}")
                    record["status"] = "Corrupted"
            else:
                print(f"Missing results for: {folder_name}")

            results_data.append(record)
            processed_count += 1

    # 5. Export to CSV
    df = pd.DataFrame(results_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved raw results to: {OUTPUT_CSV}")

    # 6. Create Analysis Pivot (Model vs Task Accuracy)
    # We combine dataset and subset for the pivot index to make it readable
    df["full_task_name"] = df.apply(
        lambda x: f"{x['dataset_name']} ({x['subset']})" if pd.notna(x["subset"]) else x["dataset_name"], axis=1
    )

    pivot_df = df.pivot_table(index="full_task_name", columns="model_name", values="eval_accuracy")
    pivot_df.to_csv(OUTPUT_PIVOT)
    print(f"Saved pivot table to: {OUTPUT_PIVOT}")

    # 7. Quick Summary: Best Model per Task
    print("\n=== Best Models (by Eval Accuracy) ===")
    valid_df = df.dropna(subset=["eval_accuracy"])
    if not valid_df.empty:
        idx = valid_df.groupby("full_task_name")["eval_accuracy"].idxmax()
        best_models = df.loc[idx][["full_task_name", "model_name", "eval_accuracy"]]
        print(best_models.to_string(index=False))
        best_models.to_csv("best_models_summary.csv", index=False)
    else:
        print("No valid accuracy data found.")


if __name__ == "__main__":
    aggregate_results()
