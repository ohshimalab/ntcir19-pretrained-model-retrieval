import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).parent.parent.parent / ".env")


client = OpenAI()


# 1. Define the Schema using Pydantic
# The order matters! We put 'analysis' first to force Chain-of-Thought.
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


def generate_instruction_structured(examples):
    """
    Induces a task instruction using Structured Outputs.
    """

    # Format your examples as a simple text block
    formatted_examples = ""
    for i, (text, label) in enumerate(examples):
        clean_text = text[:300] + "..." if len(text) > 300 else text
        formatted_examples += f'Example {i+1}:\nInput: "{clean_text}"\nLabel: "{label}"\n\n'

    # The System Prompt
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
        print(response)
        result = response.choices[0].message.parsed

        return result

    except Exception as e:
        print(f"Error: {e}")
        return None


# --- Usage Example ---
dummy_examples = [
    ("The patient denies chest pain but reports shortness of breath.", "Symptom"),
    ("Prescribed 50mg of Atenolol daily.", "Medication"),
    ("Scheduled follow-up for next Tuesday.", "Procedure"),
]

task_obj = generate_instruction_structured(dummy_examples)

if task_obj:
    print(f"--- {task_obj.task_name} ---")
    print(f"Analysis: {task_obj.analysis}")
    print(f"Instruction: {task_obj.instruction}")
