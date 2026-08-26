import json


def load_vocab(vocab_path: str) -> dict[int, str]:
    """
    Load a vocabulary file and return a mapping from token id to token string.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    return {int(token_id): token for token, token_id in vocab_data.items()}


def get_valid_next_tokens(
    partial_text: str,
    vocab: dict[int, str],
) -> list[int]:
    """
    Return token ids that can still produce a valid JSON-like output.
    This is a simplified structural filter.
    """
    valid_tokens: list[int] = []

    for token_id, token in vocab.items():
        candidate = partial_text + token

        # Very simple structural checks.
        # We keep only tokens that do not immediately break JSON structure.
        if "```" in candidate:
            continue
        if candidate.count("{") < candidate.count("}"):
            continue
        if candidate.count("[") < candidate.count("]"):
            continue
        valid_tokens.append(token_id)

    return valid_tokens


def pick_best_token(
    logits: list[float],
    valid_token_ids: list[int],
) -> int:
    """
    Pick the token id with the highest logit among the valid candidates.
    """
    if not valid_token_ids:
        raise ValueError("No valid tokens available.")

    best_token_id = valid_token_ids[0]
    best_score = logits[best_token_id]

    for token_id in valid_token_ids[1:]:
        score = logits[token_id]
        if score > best_score:
            best_score = score
            best_token_id = token_id

    return best_token_id
