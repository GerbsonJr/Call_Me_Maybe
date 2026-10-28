from llm_sdk import Small_LLM_Model
from .models import OutputItem
from .io_utils import (
    load_functions_definition, load_input_prompts, save_results)
from .decoder import ConstrainedDecoder, decode_function_call


def run_app(functions_path: str, input_path: str, output_path: str) -> int:
    """Load the inputs, decode every prompt and save the results.

    A prompt that fails is reported and skipped, so one bad prompt does
    not discard the results of the others.
    """
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
        decoder = ConstrainedDecoder(model)
    except Exception as exc:
        print(f"Error: failed to initialize model: {exc}")
        return 2

    results: list[OutputItem] = []
    failed = 0
    for idx, item in enumerate(prompts, start=1):
        print("\033[1;34m" + "─" * 60 + "\033[0m")
        print(f"\033[1;36m📨 Reading prompt {idx}/{len(prompts)}\033[0m")
        print(f"\033[1;33m→ Content:\033[0m {item.prompt}\n")
        try:
            call = decode_function_call(
                model=model,
                user_prompt=item.prompt,
                functions=functions,
                decoder=decoder,
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
        except Exception as exc:
            failed += 1
            print(
                "\033[1;31m✖ Error while processing prompt:"
                "\033[0m", item.prompt)
            print("\033[1;31mReason:\033[0m", exc)
            print("\033[1;34m" + "═" * 70 + "\033[0m")
            continue

    if failed:
        print(f"{failed} of {len(prompts)} prompts failed and were skipped.")

    return 0 if save_results(output_path, results) else 2
