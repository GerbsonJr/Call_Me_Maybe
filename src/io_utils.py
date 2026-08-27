import json
import os
from typing import Any
from pydantic import ValidationError
from .models import FunctionDefinition, InputItem, OutputItem


def load_json(path: str) -> Any | None:
    """Load JSON from a file path and handle common errors gracefully."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: file not found: {path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: invalid JSON: {path}")
        return None
    except OSError as exc:
        print(f"Error: cannot read {path}: {exc}")
        return None


def load_functions_definition(path: str) -> list[FunctionDefinition] | None:
    """Load and validate function definitions from JSON."""
    data = load_json(path)
    if not isinstance(data, list):
        print("Error: functions_definition.json must be a JSON array.")
        return None
    try:
        return [FunctionDefinition.model_validate(item) for item in data]
    except ValidationError as exc:
        print(f"Error: invalid function definition schema: {exc}")
        return None


def load_input_prompts(path: str) -> list[InputItem] | None:
    """Load and validate prompts from JSON."""
    data = load_json(path)
    if not isinstance(data, list):
        print("Error: function_calling_tests.json must be a JSON array.")
        return None
    try:
        return [InputItem.model_validate(item) for item in data]
    except ValidationError as exc:
        print(f"Error: invalid input schema: {exc}")
        return None


def save_results(path: str, results: list[OutputItem]) -> bool:
    """Save results to JSON file safely."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = [item.model_dump() for item in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return True
    except OSError as exc:
        print(f"Error: cannot write {path}: {exc}")
        return False
