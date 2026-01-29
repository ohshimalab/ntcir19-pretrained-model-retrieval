from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


class TaskDefinition(BaseModel):
    analysis: str = Field(
        description="Analyze the input patterns, the label space, and the relationship between them. Identify the domain and the specific classification rule."
    )
    task_name: str = Field(
        description="A short, descriptive name for the task (e.g., 'Clinical Trial Risk Classification')."
    )
    instruction: str = Field(
        description="A precise, imperative instruction that defines the task for an AI model. Explicitly list the labels if possible."
    )


def generate_instruction_structured(client, formatted_examples):
    """
    Induces a task instruction using Structured Outputs.
    """

    system_prompt = """You are an expert NLP Data Scientist. 
Your goal is to reverse-engineer the "Instruction" that would create the provided dataset.
1. Analyze the examples to understand the pattern.
2. Determine the domain (e.g., legal, medical, email).
3. Identify the complete set of labels used.
4. Write a clear, single-sentence instruction that would allow a human or AI to perform this task."""

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",  # Use a model that supports Structured Outputs
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Here are the dataset samples:\n\n{formatted_examples}"},
            ],
            response_format=TaskDefinition,  # <--- This enforces the schema
            temperature=0.1,  # Keep it low for analysis, but non-zero can help with phrasing
        )

        # The result is already a Python object! No JSON parsing needed.
        result = response.choices[0].message.parsed

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


def main():
    from pathlib import Path

    import pandas as pd

    project_root = Path(__file__).parent.parent.parent
    task_excel_path = project_root / "bert-tasks.xlsx"
    class_label_mapping_path = project_root / "bert-tasks-class-label-mapping.xlsx"

    df_tasks = pd.read_excel(task_excel_path).fillna("")

    # def get_task_name(row):
    #     if row["subset"]:
    #         return f"{row['dataset_name']}@{row['subset']}"
    #     return row["dataset_name"]

    # df_tasks["task_name"] = df_tasks.apply(get_task_name, axis=1)

    # def get_task_data_dir(task_name: str) -> Path:
    #     return project_root / "bert-data" / task_name.replace("/", "_")

    # df_tasks["data_dir"] = df_tasks["task_name"].apply(get_task_data_dir)

    # def get_train_dataset(data_dir: Path):
    #     df_train = pd.read_json(data_dir / "train.jsonl", lines=True)
    #     return df_train

    # def determine_integer_labels(df):
    #     labels_col_type = df["labels"].dtype
    #     if pd.api.types.is_integer_dtype(labels_col_type):
    #         return True
    #     if pd.api.types.is_object_dtype(labels_col_type):
    #         # Check if all labels can be converted to integers
    #         try:
    #             df["labels"].astype(int)
    #             return True
    #         except ValueError:
    #             return False
    #     return False

    # def get_is_integer_labels_col(task_name: str) -> bool:
    #     data_dir = get_task_data_dir(task_name)
    #     df_train = get_train_dataset(data_dir)
    #     if determine_integer_labels(df_train):
    #         return 1
    #     return 0

    # df_tasks["is_integer_labels"] = df_tasks["task_name"].apply(get_is_integer_labels_col)
    # df_tasks.to_excel(task_excel_path, index=False)

    df_class_label_mapping = pd.read_excel(class_label_mapping_path).fillna("")

    class_label_mappings = {}
    for _, row in df_class_label_mapping.iterrows():
        task_name = row["task_name"]
        label = row["label"]
        label_text = row["label_text"]
        if task_name not in class_label_mappings:
            class_label_mappings[task_name] = {}
        class_label_mappings[task_name][label] = label_text

    task_samples = {}
    for _, row in df_tasks.iterrows():
        task_name = row["task_name"]
        task_dir = row["data_dir"]
        df_train = pd.read_json(Path(task_dir) / "train.jsonl", lines=True)
        is_integer_labels = row["is_integer_labels"]
        if is_integer_labels:
            df_train["labels"] = df_train["labels"].apply(lambda x: class_label_mappings[task_name].get(x))
        if df_train["labels"].dtype == "float64":
            # round up to 3 decimal places
            df_train["labels"] = df_train["labels"].round(3)
        class_samples = {}
        unique_labels = df_train["labels"].unique()
        for label in unique_labels:
            num_samples = 5
            df_label = df_train[df_train["labels"] == label]
            if len(df_label) < num_samples:
                print(f"Warning: Task {task_name} label {label} has only {len(df_label)} samples.")
                num_samples = len(df_label)
            samples = df_label.sample(num_samples, random_state=42)
            sample_texts = samples["text"].tolist()
            truncated_texts = [text if len(text) <= 100 else text[:97] + "..." for text in sample_texts]
            class_samples[label] = truncated_texts
        task_samples[task_name] = class_samples

    task_prompts = {}
    for task_name, class_samples in task_samples.items():
        formatted_examples = ""
        current_example_index = 1
        for label, samples in class_samples.items():
            for sample in samples:
                formatted_examples += f"Example {current_example_index}:\nInput: {sample}\nLabel: {label}\n\n"
                current_example_index += 1
        task_prompts[task_name] = formatted_examples.strip()

    import json

    output_path = project_root / "task_few_shot_samples.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(task_prompts, f, ensure_ascii=False, indent=4)

    load_dotenv(project_root / ".env")

    client = OpenAI()

    first_task_prompt = next(iter(task_prompts.values()))
    result = generate_instruction_structured(client, first_task_prompt)
    print("Generated Instruction:")
    print(result.task_name)
    print(result.instruction)


if __name__ == "__main__":
    main()
