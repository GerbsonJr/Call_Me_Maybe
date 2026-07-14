import json
from typing import Any


def load_json(path: str) -> list | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return (json.load(f))
    except FileNotFoundError:
        print(f"Erro: arquivo não encontrado: {path}")
        return None
    except json.JSONDecodeError:
        print(f"Erro: JSON inválido: {path}")
        return None
    except OSError as exc:
        print(f"Erro: não foi possível ler {path}: {exc}")
        return None


def save_json(path: str, data: Any) -> bool:
    try:
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except OSError as exc:
        print(f"Erro: não foi possível escrever {path}: {exc}")
        return False


def run_app(functions_path: Any, input_path: Any, output_path: Any) -> int:

    functions = load_json(functions_path)
    inputs = load_json(input_path)
    if functions is None or inputs is None:
        print("Error: Could not load input files.")
        return 2

    if not isinstance(functions, list):
        print("Error: functions_definition.json must be a JSON array.")
        return 2

    if not isinstance(inputs, list):
        print("Error: function_calling_tests.json must be a JSON array.")
        return 2

    if len(functions) == 0:
        print("Error: No function available in functions_definition.json.")
        return 2

    first_fn = functions[0]
    if not isinstance(first_fn, dict) or "name" not in first_fn:
        print("Error: Invalid function definition (missing 'name' field).")
        return 2

    results: list[dict[str, Any]] = []
    for item in inputs:
        if not isinstance(item, dict) or "prompt" not in item:
            continue
        prompt = item["prompt"]
        if not isinstance(prompt, str):
            continue
        results.append(
            {
                "prompt": prompt,
                "name": str(first_fn["name"]),   # dummy por enquanto
                "parameters": {},                # dummy por enquanto
            }
        )
    ok = save_json(output_path, results)
    return 0 if ok else 2
