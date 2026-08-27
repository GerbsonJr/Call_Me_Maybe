from typing import Iterable
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition


def build_function_choice_prompt(
    user_prompt: str,
    functions: Iterable[FunctionDefinition],
) -> str:
    """Build the instruction prompt used to choose a function name."""
    lines: list[str] = []
    lines.append("You are a function selection engine.")
    lines.append("Choose exactly one function name from the available list.")
    lines.append("Return only the function name, no explanation.")
    lines.append("")
    lines.append("Available functions:")
    for fn_def in functions:
        param_names = ", ".join(fn_def.parameters.keys()) or "no parameters"
        lines.append(
            f"- {fn_def.name}: {fn_def.description} (params: {param_names})"
        )
    lines.append("")
    lines.append(f'User request: "{user_prompt}"')
    lines.append("Answer:")
    return "\n".join(lines)


def choose_function_name(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Choose a function name using the model output with safe fallback."""
    function_names = {fn_def.name for fn_def in functions}
    if not function_names:
        raise ValueError("No functions available for selection.")

    prompt_text = build_function_choice_prompt(
        user_prompt=user_prompt,
        functions=functions,
    )

    try:
        token_tensor = model.encode(prompt_text)
        input_ids = token_tensor.tolist()

        if (
            isinstance(input_ids, list) and input_ids
                and isinstance(input_ids[0], list)):

            input_ids = input_ids[0]

        logits = model.get_logits_from_input_ids(input_ids)

        # NOTE:
        # This is a temporary/simple stage. We do NOT decode from logits yet.
        # We use a conservative fallback until constrained decoding is implemented.
        _ = logits  # keep explicit usage for lint clarity

        # Temporary fallback: first function (will be replaced by constrained decoding)
        return functions[0].name

    except Exception as exc:  # pylint: disable=broad-except
        print(
            f"Warning: model selection failed, using fallback. Details: {exc}")
        return functions[0].name
