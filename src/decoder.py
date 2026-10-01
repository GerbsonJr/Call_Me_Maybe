"""Constrained decoding engine.

Implements constrained decoding as required by the subject (V.3.3): the
model's logits are masked at every generation step so that only tokens
compatible with a valid grammar remain selectable.

DESIGN NOTE ON PARAMETER VALUES (read this before touching this file):
Early versions of this module let the model generate parameter values as
free text, token by token, only checking that the *shape* stayed valid
(digits for numbers, non-quote characters for strings). In testing, this
was unreliable with the small 0.6B base model: numbers ran away into
nonsense ("2" became "2e+22"), strings leaked the model's own commentary
("shrek" became "shrek is a name that is used in..."), or got stuck
repeating a token forever ("hello" became "hellohellohello...").

The fix is NOT to keep tuning the free-generation stopping heuristic — a
tiny, non-instruction-tuned model simply cannot reliably reproduce an
exact number or word from scratch, character by character. Instead,
candidate values are enumerated directly from the user's prompt (numbers
present in the text, quoted substrings, individual words), and constrained
decoding is used to have the model *choose* among those candidates via a
token-level trie — the same technique already used for function-name
selection. This guarantees every produced value is an exact, verbatim
match for something that was actually in the prompt (no truncation, no
runaway repetition), while the selection *among* candidates still comes
from the model's own logits, not from string-matching heuristics.

ASSUMPTION TO VERIFY: `Small_LLM_Model.get_path_to_vocab_file()` is assumed
to return a JSON file mapping token string -> token id (the common
BPE/vocab.json format). Leading-space tokens are assumed to use a marker
such as "Ġ" (GPT-style BBPE), and newline/tab tokens "Ċ"/"ĉ" respectively;
adjust the marker tables below if your tokenizer differs.
"""

import json
import math
import re
from typing import Any, Callable, Optional

from llm_sdk import Small_LLM_Model
from .models import FunctionDefinition


SPACE_MARKERS = ("Ġ", "▁")
NEWLINE_MARKERS = {"Ċ": "\n", "ĉ": "\t"}
NUMBER_PARTIAL = re.compile(r"-?\d*(\.\d*)?$")
NUMBER_COMPLETE = re.compile(r"-?\d+(\.\d+)?$")
STRING_STOP_CHARS = ('"', "'", "{", "}", "[", "]", "\n", "\t")
BOOLEAN_OPTIONS = ("true", "false")

NUMBER_CAPTURE = re.compile(r"-?\d+(?:\.\d+)?")
QUOTE_CAPTURE = re.compile(r"'([^']*)'|\"([^\"]*)\"")
WORD_CAPTURE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")


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
    for marker, real_char in NEWLINE_MARKERS.items():
        token_str = token_str.replace(marker, real_char)
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
        probability is selected. Stops when `is_complete` is True, no valid
        token remains, or `max_new_tokens` is reached.
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


def select_from_candidates(
    decoder: ConstrainedDecoder,
    prompt_ids: list[int],
    candidates: list[str],
    max_new_tokens: int = 32,
) -> str:
    """Pick one of a fixed set of exact
    candidate strings via constrained decoding.

    A token is only accepted while the accumulated text is a prefix of at
    least one candidate. This is the same trie-based technique used for
    function-name selection, generalized to any closed set of exact
    strings: it guarantees the result is always one of the candidates
    verbatim (no truncation, no drift), while the choice among them still
    comes from the model's own logits.
    """
    if not candidates:
        return ""
    if len(candidates) == 1:
        return candidates[0]

    def is_valid(current: str, candidate: str) -> bool:
        trial = (current + candidate).strip()
        return trial == "" or any(c.startswith(trial) for c in candidates)

    def is_complete(current: str) -> bool:
        return current.strip() in candidates

    result = decoder.generate(
        input_ids=prompt_ids,
        is_valid_continuation=is_valid,
        is_complete=is_complete,
        max_new_tokens=max_new_tokens,
    ).strip()

    return result if result in candidates else candidates[0]


def choose_function_name(
    decoder: ConstrainedDecoder,
    prompt_ids: list[int],
    functions: list[FunctionDefinition],
) -> str:
    """Select a function name via trie-constrained
    decoding over the LLM's logits."""
    names = [fn.name for fn in functions]
    result = select_from_candidates(
        decoder, prompt_ids, names, max_new_tokens=16)
    if result not in names:
        raise ValueError(f"Could not select a valid function name: {result!r}")
    return result


def _find_number_candidates(text: str) -> list[str]:
    """Return every numeric substring that literally appears in text, in
    order, WITHOUT deduplicating — if the same number appears twice (e.g.
    "sum of 0 and 0"), both occurrences must remain available so each
    parameter can claim its own, rather than the second one falling
    through to less reliable free-form generation."""
    return [match.group(0) for match in NUMBER_CAPTURE.finditer(text)]


def _find_quoted_candidates(text: str) -> list[str]:
    """Return substrings enclosed in single or double quotes in text.
    An empty pair of quotes ('' or "") yields an empty-string candidate —
    this matters because "reverse the string ''" must resolve to "",
    not silently skip to some other word in the prompt."""
    results: list[str] = []
    for match in QUOTE_CAPTURE.finditer(text):
        value = (match.group(1) if match.group(1)
                 is not None else match.group(2))
        if value is not None and value not in results:
            results.append(value)
    return results


def _find_word_candidates(text: str) -> list[str]:
    """Return the distinct alphabetic words that appear in text, in order."""
    seen: list[str] = []
    for match in WORD_CAPTURE.finditer(text):
        word = match.group(0)
        if word not in seen:
            seen.append(word)
    return seen


def _sort_by_appearance(text: str, raw_values: list[str]) -> list[str]:
    """Order raw numeric substrings by where they first appear in text.

    The model decides WHICH numbers are relevant for the call (correctly
    ignoring distractors, e.g. a stated age mixed in with the actual sum);
    this just maps the already-chosen values back onto parameters in the
    order they were written, matching the convention used throughout the
    subject's own examples (first number -> first parameter)."""
    return sorted(raw_values, key=lambda v: text.find(v))


def is_valid(current: str, candidate: str) -> bool:
    trial = (current + candidate).strip()
    if NUMBER_PARTIAL.fullmatch(trial):
        return True
    return (bool(NUMBER_COMPLETE.fullmatch(current.strip()))
            and candidate.strip() == "")


def is_complete(current: str) -> bool:
    stripped = current.strip()
    return bool(NUMBER_COMPLETE.fullmatch(stripped)) and current != stripped


def generate_string_parameter(
    decoder: ConstrainedDecoder,
    prompt_ids: list[int],
    max_new_tokens: int = 8,
) -> str:
    """Fallback free-form string generation, used only when the prompt has
    no quoted text and no words for the model to select from."""

    def is_valid(current: str, candidate: str) -> bool:
        return not any(ch in candidate for ch in STRING_STOP_CHARS)

    def is_complete(current: str) -> bool:
        return current.strip() != "" and current[-1].isspace()

    raw = decoder.generate(
        input_ids=prompt_ids,
        is_valid_continuation=is_valid,
        is_complete=is_complete,
        max_new_tokens=max_new_tokens,
    )
    return raw.strip()


def generate_boolean_parameter(
    decoder: ConstrainedDecoder,
    prompt_ids: list[int],
) -> bool:
    """Select a boolean value via
    trie-constrained decoding over {true, false}."""
    result = select_from_candidates(
        decoder, prompt_ids, list(BOOLEAN_OPTIONS), max_new_tokens=4
    )
    return result == "true"


def _build_param_prompt(
    user_prompt: str,
    fn_name: str,
    param_name: str,
    param_type: str,
    already_filled: dict[str, Any],
) -> str:
    """Build the instruction prompt used
    to select/generate a parameter value."""
    filled_line = ""
    if already_filled:
        pairs = ", ".join(f"{k}={v!r}" for k, v in already_filled.items())
        filled_line = f"Already assigned parameters: {pairs}\n"
    return (
        f'User request: "{user_prompt}"\n'
        f"Selected function: {fn_name}\n"
        f"{filled_line}"
        f"Provide the value for parameter"
        f" '{param_name}' (type: {param_type}).\n"
        f"Value:"
    )


def decode_function_call(
    model: Small_LLM_Model,
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> dict[str, Any]:
    """Decode a full function call using constrained decoding end-to-end.

    The function name and every parameter value are selected by the model
    via logit-masked decoding. Numeric and string parameters are, whenever
    possible, selected from candidates found verbatim in the prompt (see
    module docstring for why free-form character generation was dropped)
    rather than generated from scratch.
    """
    if not functions:
        return {"name": "", "parameters": {}}

    decoder = ConstrainedDecoder(model)
    fn_map = {fn.name: fn for fn in functions}

    name_prompt = (
        "Examples of how to map a request to a function name:\n"
        + "\n".join(
            f'Request about: "{fn.description}" -> Function name: {fn.name}'
            for fn in functions
        )
        + "\n\nAvailable functions:\n"
        + "\n".join(f"- {fn.name}: {fn.description}" for fn in functions)
        + f'\n\nUser request: "{user_prompt}"\nFunction name:'
    )
    chosen_name = choose_function_name(decoder,
                                       decoder.encode(name_prompt), functions)
    fn_def = fn_map[chosen_name]

    parameters: dict[str, Any] = {}
    remaining_numbers = _find_number_candidates(user_prompt)
    used_strings: set[str] = set()
    number_assignments: list[tuple[str, str]] = []

    quoted_candidates = _find_quoted_candidates(user_prompt)
    word_candidates = _find_word_candidates(user_prompt)

    for param_name, param_def in fn_def.parameters.items():
        param_prompt = _build_param_prompt(
            user_prompt, chosen_name, param_name, param_def.type, parameters
        )
        param_ids = decoder.encode(param_prompt)

        if param_def.type in ("number", "integer"):
            if remaining_numbers:
                chosen = select_from_candidates(decoder,
                                                param_ids, remaining_numbers)
                remaining_numbers.remove(chosen)  # removes only ONE occurrence
                number_assignments.append((param_name, chosen))
                parameters[param_name] = float(chosen)
            else:
                value = generate_number_parameter(decoder, param_ids)
                parameters[param_name] = (
                    int(value) if param_def.type == "integer" else value
                )

        elif param_def.type == "string":
            if quoted_candidates:
                candidates = [c for c in quoted_candidates
                              if c not in used_strings]
            else:
                candidates = [c for c in word_candidates
                              if c not in used_strings]
            if candidates:
                chosen = select_from_candidates(decoder, param_ids, candidates)
                used_strings.add(chosen)
                parameters[param_name] = chosen
            else:
                parameters[param_name] = generate_string_parameter(decoder,
                                                                   param_ids)

        elif param_def.type == "boolean":
            parameters[param_name] = generate_boolean_parameter(decoder,
                                                                param_ids)

        else:
            parameters[param_name] = None

    if len(number_assignments) >= 2:
        raw_values = [raw for _, raw in number_assignments]
        ordered_raw = _sort_by_appearance(user_prompt, raw_values)
        for (param_name, _), raw in zip(number_assignments, ordered_raw):
            param_def = fn_def.parameters[param_name]
            value = float(raw)
            parameters[param_name] = (
                int(value) if param_def.type == "integer" else value
            )

    return {"name": chosen_name, "parameters": parameters}
