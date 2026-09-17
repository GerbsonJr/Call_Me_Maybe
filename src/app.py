from llm_sdk import Small_LLM_Model
from .models import OutputItem
from .io_utils import (
    load_functions_definition, load_input_prompts, save_results)
from .decoder import decode_function_call
from time import sleep


def run_app(functions_path: str, input_path: str, output_path: str) -> int:
    functions = load_functions_definition(functions_path)
    prompts = load_input_prompts(input_path)

    if functions is None or prompts is None:
        print("Error: could not load input files.")
        return 2
    if len(functions) == 0:
        print("Error: no functions available.")
        return 2

    try:
        model = Small_LLM_Model()
    except Exception as exc:
        print(f"Error: failed to initialize model: {exc}")
        return 2

    results: list[OutputItem] = []
    for idx, item in enumerate(prompts, start=1):
        print("\033[1;34m" + "─" * 60 + "\033[0m")
        print(f"\033[1;36m📨 Reading prompt {idx}/{len(prompts)}\033[0m")
        print(f"\033[1;33m→ Content:\033[0m {item.prompt}\n")
        try:
            call = decode_function_call(
                model=model,
                user_prompt=item.prompt,
                functions=functions,
            )
            output = OutputItem(
                prompt=item.prompt,
                name=call["name"],
                parameters=call["parameters"],
            )
            results.append(output)

            print("\033[1;32m✔ Function detected:\033[0m", output.name)
            print("\033[1;32m✔ Parameters:\033[0m", output.parameters)
            print("\033[1;34m" + "─" * 60 + "\033[0m\n")

            sleep(0.1)
        except Exception as exc:
            print(
                "\033[1;31m✖ Error while processing prompt:"
                "\033[0m", item.prompt)
            print("\033[1;31mReason:\033[0m", exc)
            print("\033[1;34m" + "═" * 70 + "\033[0m")
            return 2

    return 0 if save_results(output_path, results) else 2
