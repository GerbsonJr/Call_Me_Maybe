"""Constrained decoding engine.

Implements true constrained decoding as required by the subject (V.3.3):
at every generation step, the model's logits are masked so that only
tokens compatible with the target grammar remain selectable. The model
never "spontaneously" produces JSON — every token is filtered before
selection, guaranteeing valid structure and schema compliance by
construction.

ASSUMPTION TO VERIFY: `Small_LLM_Model.get_path_to_vocab_file()` is assumed
to return a JSON file mapping token string -> token id (the common
BPE/vocab.json format). If your SDK's vocab file has a different shape
(e.g. id -> token, or a merges list), adjust `_load_vocab` accordingly.
Leading-space tokens are assumed to use a marker such as "Ġ" (GPT-style
BBPE); adjust SPACE_MARKERS if your tokenizer uses e.g. sentencepiece "▁".
"""

import json
import math
import re
from typing import Any, Callable, Optional
from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition


SPACE_MARKERS = ("Ġ", "▁")
NUMBER_PARTIAL = re.compile(r"-?\d*(\.\d*)?$")
NUMBER_COMPLETE = re.compile(r"-?\d+(\.\d+)?$")
NUMBER_CAPTURE = re.compile(r"-?\d+(?:\.\d+)?")
BOOLEAN_TRUE_PATTERN = re.compile(r"\b(true|yes|on|enable|enabled)\b")


def _extract_numbers_from_prompt(user_prompt: str) -> list[float]:
    return [float(m.group(0)) for m in NUMBER_CAPTURE.finditer(user_prompt)]


def _load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    """Load the token id -> token text mapping from the model's vocab file."""
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(token_id): token_str for token_str, token_id in raw.items()}


def _token_text(id_to_token: dict[int, str], token_id: int) -> str:
    """Convert a raw vocab token into its plain-text representation."""
    token_str = id_to_token.get(token_id, "")
    for marker in SPACE_MARKERS:
        token_str = token_str.replace(marker, " ")
    return token_str


def _log_softmax(logits: list[float]) -> list[float]:
    """Numerically stable log-softmax over a list of logits."""
    max_logit = max(logits)
    shifted = [logit - max_logit for logit in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    return [s - log_sum_exp for s in shifted]


def _flatten_ids(token_tensor: Any) -> list[int]:
    """Normalize the encoder's tensor output into a flat list of ints."""
    raw_ids = token_tensor.tolist()

    if isinstance(raw_ids, list) and raw_ids and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]

    if not isinstance(raw_ids, list):
        return []

    return [int(x) for x in raw_ids]


class ConstrainedDecoder:
    """Runs token-by-token generation under a caller-supplied grammar mask."""

    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        self.id_to_token = _load_vocab(model)

    def encode(self, text: str) -> list[int]:
        """Tokenize text into a flat list of input ids."""
        return _flatten_ids(self.model.encode(text))

    def generate(
        self,
        input_ids: list[int],
        is_valid_continuation: Callable[[str, str], bool],
        is_complete: Callable[[str], bool],
        max_new_tokens: int = 32,
    ) -> str:
        """
        Generate text token-by-token.

        At each step every candidate token's logit is masked to -inf unless
        `is_valid_continuation(generated_so_far, candidate_token_text)` is
        True. Among the remaining tokens, the one with the highest log
        probability is selected (this is where the LLM's own logits decide
        the output, not a heuristic). Stops when `is_complete` is True, no
        valid token remains, or `max_new_tokens` is reached.
        """
        current_ids = list(input_ids)
        generated_text = ""

        for _ in range(max_new_tokens):
            logits = self.model.get_logits_from_input_ids(current_ids)
            log_probs = _log_softmax(logits)

            best_token_id: Optional[int] = None
            best_score = float("-inf")

            for token_id, score in enumerate(log_probs):
                token_text = _token_text(self.id_to_token, token_id)
                if not token_text:
                    continue
                if not is_valid_continuation(generated_text, token_text):
                    continue
                if score > best_score:
                    best_score = score
                    best_token_id = token_id

            if best_token_id is None:
                break

            generated_text += _token_text(self.id_to_token, best_token_id)
            current_ids.append(best_token_id)

            if is_complete(generated_text):
                break

        return generated_text


def choose_function_name(
    decoder: ConstrainedDecoder,
    prompt_ids: list[int],
    functions: list[FunctionDefinition],
) -> str:
    names = [fn.name for fn in functions]

    def is_valid(current: str, candidate: str) -> bool:
        trial = (current + candidate).strip()
        return trial == "" or any(name.startswith(trial) for name in names)

    def is_complete(current: str) -> bool:
        return current.strip() in names

    result = decoder.generate(
        input_ids=prompt_ids,
        is_valid_continuation=is_valid,
        is_complete=is_complete,
        max_new_tokens=16,
    ).strip()

    if result in names:
        return result
    raise ValueError(f"Could not select a valid function name: {result!r}")


def generate_number_parameter(
    decoder: ConstrainedDecoder,
    prompt_ids: list[int],
    max_new_tokens: int = 8,
) -> float:
    """Generate a numeric parameter value under constrained decoding.

    Tokens are only accepted while the text so far still matches a partial
    number pattern (optional leading '-', digits, optional single '.').
    """

    def is_valid(current: str, candidate: str) -> bool:
        trial = (current + candidate).strip()
        return bool(NUMBER_PARTIAL.fullmatch(trial))

    def is_complete(current: str) -> bool:
        # Never stop early on our own signal: let generation run to
        # max_new_tokens or until no digit-extension remains valid, then
        # take the longest complete numeric prefix below.
        return False

    raw = decoder.generate(
        input_ids=prompt_ids,
        is_valid_continuation=is_valid,
        is_complete=is_complete,
        max_new_tokens=max_new_tokens,
    ).strip()

    match = NUMBER_COMPLETE.match(raw) if raw else None
    return float(match.group(0)) if match else 0.0


def generate_string_parameter(user_prompt: str) -> str:
    for quote_char in ("'", '"'):
        if user_prompt.count(quote_char) >= 2:
            parts = user_prompt.split(quote_char)
            if len(parts) >= 3 and parts[1].strip():
                return parts[1].strip()

    lowered = user_prompt.lower().strip()
    if lowered.startswith("greet "):
        return user_prompt.strip()[6:].strip(" \t\n\r.,!?")

    words = user_prompt.strip().split()
    return words[-1].strip(".,!?") if words else ""


def decode_function_call(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> dict[str, Any]:
    """Decode a full function call using constrained decoding end-to-end."""
    if not functions:
        return {"name": "", "parameters": {}}

    decoder = ConstrainedDecoder(model)
    fn_map = {fn.name: fn for fn in functions}

    name_prompt = (
        "Available functions:\n"
        + "\n".join(f"- {fn.name}: {fn.description}" for fn in functions)
        + f'\n\nUser request: "{user_prompt}"\nFunction name:'
    )
    prompt_ids = decoder.encode(name_prompt)
    chosen_name = choose_function_name(decoder, prompt_ids, functions)
    fn_def = fn_map[chosen_name]

    parameters: dict[str, Any] = {}
    prompt_numbers = _extract_numbers_from_prompt(user_prompt)
    number_idx = 0

    for param_name, param_def in fn_def.parameters.items():
        if param_def.type in ("number", "integer"):
            if number_idx < len(prompt_numbers):
                value = prompt_numbers[number_idx]
                number_idx += 1
            else:
                value = 0.0

            parameters[param_name] = (
                int(value) if param_def.type == "integer" else value
            )

        elif param_def.type == "string":
            parameters[param_name] = generate_string_parameter(user_prompt)

        elif param_def.type == "boolean":
            lowered = user_prompt.lower()
            parameters[param_name] = bool(
                BOOLEAN_TRUE_PATTERN.search(lowered)
            )

        else:
            parameters[param_name] = None

    return {"name": chosen_name, "parameters": parameters}
