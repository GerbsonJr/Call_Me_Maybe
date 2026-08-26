from llm_sdk.llm_sdk import Small_LLM_Model
from .models import OutputItem
from .io_utils import load_functions_definition, load_input_prompts, save_results
from .decoder import decode_function_call


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
    for item in prompts:
        call = decode_function_call(
            model=model,
            user_prompt=item.prompt,
            functions=functions,
        )
        results.append(
            OutputItem(
                prompt=item.prompt,
                name=call["name"],
                parameters=call["parameters"],
            )
        )

    return 0 if save_results(output_path, results) else 2
