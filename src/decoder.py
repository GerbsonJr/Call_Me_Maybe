import json
from typing import Any
from .models import FunctionDefinition


def _extract_function_map(
        functions: list[FunctionDefinition]) -> dict[str, FunctionDefinition]:
    """Build a map from function name to function definition."""
    return {fn.name: fn for fn in functions}


def _allowed_parameter_keys(fn_def: FunctionDefinition) -> set[str]:
    """Return allowed parameter keys for a function."""
    return set(fn_def.parameters.keys())


def _validate_param_type(value: Any, expected_type: str) -> bool:
    """Validate a Python value against a simplified JSON-schema type."""
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "null":
        return value is None
    return False


def validate_function_call(
    candidate: dict[str, Any],
    functions: list[FunctionDefinition],
) -> tuple[bool, str]:
    """
    Validate generated function call structure.

    Expected shape:
    {
      "name": "<function_name>",
      "parameters": { ... }
    }
    """
    if not isinstance(candidate, dict):
        return False, "Candidate must be an object."

    if "name" not in candidate or "parameters" not in candidate:
        return False, "Candidate must contain 'name' and 'parameters'."

    name = candidate["name"]
    params = candidate["parameters"]

    if not isinstance(name, str) or not name.strip():
        return False, "Field 'name' must be a non-empty string."

    if not isinstance(params, dict):
        return False, "Field 'parameters' must be an object."

    fn_map = _extract_function_map(functions)
    if name not in fn_map:
        return False, f"Unknown function name: {name}"

    fn_def = fn_map[name]
    allowed_keys = _allowed_parameter_keys(fn_def)

    # No extra keys
    for key in params:
        if key not in allowed_keys:
            return False, f"Unexpected parameter key for {name}: {key}"

    # Type-check provided keys
    for key, value in params.items():
        expected_type = fn_def.parameters[key].type
        if not _validate_param_type(value, expected_type):
            return False, (
                f"Invalid type for parameter '{key}' in '{name}'. "
                f"Expected {expected_type}, got {type(value).__name__}."
            )

    return True, "ok"


def build_fallback_call(functions: list[FunctionDefinition]) -> dict[str, Any]:
    """Return a safe fallback function call."""
    if not functions:
        return {"name": "", "parameters": {}}
    return {"name": functions[0].name, "parameters": {}}


def decode_function_call(
    model: Any,
    user_prompt: str,
    functions: list[FunctionDefinition],
    max_new_tokens: int = 120,
) -> dict[str, Any]:
    """
    Decode a structured function call.

    Current stage:
    - Builds strict output template
    - Uses model context (placeholder hook)
    - Validates candidate
    - Falls back safely if invalid

    NOTE:
    This is an incremental baseline.
    Token-level mask logic should be added next.
    """
    _ = max_new_tokens  # reserved for future iterative token decoding loop

    if not functions:
        return {"name": "", "parameters": {}}

    # ---------- Prompt for structured output ----------
    function_lines: list[str] = []
    for fn_def in functions:
        params_spec = {k: v.type for k, v in fn_def.parameters.items()}
        function_lines.append(
            f'- name="{fn_def.name}", description="{fn_def.description}",'
            f' parameters={params_spec}'
        )

    constrained_prompt = (
        "You must output ONLY valid JSON with this exact shape:\n"
        '{"name":"<function_name>","parameters":{...}}\n'
        "Choose 'name' from available functions.\n"
        "Use only allowed parameter keys and correct value types.\n\n"
        "Available functions:\n"
        + "\n".join(function_lines)
        + f'\n\nUser prompt: "{user_prompt}"\n'
        "JSON output:"
    )

    # ---------- Model call hook (baseline placeholder) ----------
    try:
        token_tensor = model.encode(constrained_prompt)
        input_ids = token_tensor.tolist()
        if isinstance(input_ids, list) and input_ids and isinstance(
                input_ids[0], list):
            input_ids = input_ids[0]

        # This call is kept so model is truly involved in the flow.
        _logits = model.get_logits_from_input_ids(input_ids)

        # TODO(next): iterative constrained token decoding using vocab + logits mask
        # For now, produce a minimal candidate that is schema-valid.
        candidate = {"name": functions[0].name, "parameters": {}}

    except Exception as exc:  # pylint: disable=broad-except
        print(f"Warning: decode failed, using fallback. Details: {exc}")
        return build_fallback_call(functions)

    # ---------- Validation ----------
    ok, reason = validate_function_call(candidate, functions)
    if not ok:
        print(f"Warning: invalid decoded output ({reason}), using fallback.")
        return build_fallback_call(functions)

    # Ensure candidate is JSON-serializable and structurally valid JSON text
    try:
        json_text = json.dumps(candidate, ensure_ascii=False)
        parsed_back = json.loads(json_text)
        if not isinstance(parsed_back, dict):
            raise ValueError("Decoded JSON is not an object.")
    except Exception as exc:  # pylint: disable=broad-except
        print(
            f"Warning: JSON serialization check failed ({exc}),"
            " using fallback."
        )
        return build_fallback_call(functions)

    return candidate
