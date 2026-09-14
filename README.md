# Call Me Maybe

*This project has been created as part of the 42 curriculum by gmateus-.*

## Description

**Call Me Maybe** is a function-calling engine that translates natural-language
requests into structured, machine-executable function calls, using a small
(0.6B parameter) language model — Qwen3-0.6B.

Given a prompt such as *"What is the sum of 40 and 2?"*, a traditional LLM
would simply answer *"42"* in plain text. This project instead outputs a
structured object describing **which function to call and with what
arguments**:

```json
{
  "name": "fn_add_numbers",
  "parameters": {"a": 40, "b": 2}
}
```

Small language models are notoriously unreliable at spontaneously producing
valid JSON — success rates as low as 30% are common when relying on
prompting alone. The core challenge (and the actual subject of this project)
is not the arithmetic or the language understanding: it is guaranteeing
**100% structurally and semantically valid output** from a model that,
left to its own devices, cannot be trusted to produce it. The solution
implemented here is **constrained decoding**: the model's raw output
probabilities (logits) are intercepted and masked at every generation step,
so that only tokens compatible with a valid function name and the target
JSON schema can ever be selected.

## Instructions

### Requirements
- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- The `llm_sdk` package available alongside `src/` (see project layout)

### Installation

```bash
make install
```

This resolves and installs dependencies (`pydantic`, `numpy`) via `uv`.
The grading environment runs `uv sync` directly, which relies on the same
`pyproject.toml`.

### Running

```bash
make run
```

or directly:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All three arguments are optional; by default the program reads from
`data/input/` and writes to `data/output/`.

### Other Makefile targets

| Target | Purpose |
|---|---|
| `make install` | Install dependencies |
| `make run` | Run the pipeline with default paths |
| `make debug` | Run under `pdb` |
| `make clean` | Remove caches (`__pycache__`, `.mypy_cache`, etc.) |
| `make lint` | Run `flake8` and `mypy` with the required flags |

## Algorithm Explanation

Constrained decoding is implemented in `decoder.py` around a single
primitive, `ConstrainedDecoder.generate`, which drives a token-by-token
generation loop:

1. At each step, `get_logits_from_input_ids` is called on the tokens
   generated so far.
2. Every token in the vocabulary is decoded to its text form (using the
   vocab file returned by `get_path_to_vocab_file`), and a caller-supplied
   `is_valid_continuation(current_text, candidate_token_text)` predicate
   decides whether appending it could still lead to a valid output.
   Invalid tokens are effectively excluded from selection (equivalent to
   setting their logit to `-inf`).
3. Among the remaining valid tokens, the one with the highest log
   probability (via a numerically stable log-softmax over the logits) is
   selected — so the LLM's own confidence, not a heuristic, decides the
   output *within* the allowed grammar.
4. Generation stops once a caller-supplied `is_complete(current_text)`
   predicate is satisfied, no valid token remains, or a maximum token
   budget is reached.

This single primitive is reused for two different grammars:

- **Function name selection** (`choose_function_name`): the grammar is a
  trie over the set of known function names. A candidate continuation is
  valid only if it is a prefix of at least one function name; generation
  is complete once the accumulated text exactly matches one. This directly
  satisfies the requirement that the function be chosen *by the model*,
  not by string-matching heuristics.
- **Numeric parameter generation** (`generate_number_parameter`): the
  grammar accepts only text matching a partial number pattern
  (`-?\d*(\.\d*)?`), guaranteeing that whatever is generated can always be
  parsed as a valid `number`/`integer` per the schema.

String-typed parameters are currently extracted directly from the prompt
text (quoted substrings, or the trailing word for greeting-style prompts)
rather than generated token-by-token — see **Challenges Faced** below for
why, and **Design Decisions** for the trade-off this represents.

## Design Decisions

- **Grammar as a predicate, not a fixed automaton.** Rather than building a
  full formal grammar/parser for JSON, `ConstrainedDecoder.generate` takes
  two plain predicates (`is_valid_continuation`, `is_complete`) supplied by
  the caller. This keeps the decoding loop itself small and testable, and
  lets each field (function name, number, string) define its own notion of
  "valid so far" independently.
- **Pydantic for all schema validation** (`models.py`): `FunctionDefinition`,
  `InputItem`, and `OutputItem` are validated on load and on write, so
  malformed input data or an internal bug producing an inconsistent object
  fails loudly and early rather than silently producing bad output.
- **Per-prompt fault isolation** (`app.py`): decoding is wrapped in a
  `try/except` *inside* the loop over prompts. A failure on one prompt logs
  a warning and falls back to a safe default rather than aborting the
  entire run — important given the requirement to process potentially
  hundreds of prompts without crashing.
- **Vocab format assumption.** The vocab file is assumed to map
  `{token_string: token_id}` (the common BPE/`vocab.json` shape), with
  leading-space tokens marked by a special character (e.g. `Ġ`). This was
  confirmed against the model's own vocab file during development.

## Performance Analysis

> _Fill in with your actual measured numbers before submission — run
> `make run` on the full test set and record the results here._

- **Accuracy:** function-name selection accuracy and parameter-extraction
  correctness, measured against `_______` labeled/expected prompts:
  `___%`.
- **Validity:** JSON validity rate: `100%` by construction — every output
  passes through Pydantic validation (`OutputItem`) before being written,
  and the constrained decoding loop cannot produce a token that breaks the
  target grammar.
- **Speed:** total time to process the full `function_calling_tests.json`
  set: `___` seconds (target: under 5 minutes per the subject).
- **Known bottleneck:** each generation step scores the entire vocabulary
  to find valid continuations, which is the dominant cost. This is
  acceptable at the scale required here but would need batching/caching to
  scale further.

## Challenges Faced

- **Distinguishing real constrained decoding from decoration.** An early
  version of the decoding logic computed logits but never used them to
  restrict token choice, and used `model.decode(model.encode(prompt))` —
  which just re-encodes/decodes the prompt itself rather than generating
  anything new. This produced output that looked plausible but was never
  actually driven by the model. It was caught by tracing exactly which
  variable fed into the final JSON and confirming logits were discarded
  (`_ = model.get_logits_from_input_ids(...)`).
- **Avoiding heuristic function selection.** It is tempting to resolve
  simple cases with string matching (e.g. `"greet" in prompt`), but the
  subject explicitly forbids this. The trie-based constrained decoding
  approach for function names was built specifically to avoid any
  heuristic shortcut while still guaranteeing a valid name.
- **Open-ended string parameters.** Constraining arbitrary free text at the
  character level (unlike a fixed set of function names or a number
  pattern) would require a much larger grammar to remain meaningful; as a
  practical trade-off for this project's scope, string values are
  extracted from the prompt directly instead of generated by the model.
- **Duplicated responsibility across modules.** An intermediate version had
  function-selection logic split across `llm_engine.py` and `decoder.py`.
  Consolidating decoding into a single `ConstrainedDecoder` avoided
  divergent behavior between the two.

## Testing Strategy

- Unit tests (not graded, used for local validation) cover:
  - `models.py`: schema validation accepts valid definitions and rejects
    empty names/prompts and unsupported types.
  - `io_utils.py`: missing files, invalid JSON, and valid files each
    produce the expected result without raising.
  - `decoder.py`: `choose_function_name` always returns a name from the
    provided function list; `generate_number_parameter` always returns a
    value parseable as `float`.
- Manual testing with edge cases from the subject: empty strings, large
  numbers, special characters, wrong types, ambiguous prompts, and
  multi-parameter functions.
- End-to-end testing via `make run` against the example
  `function_calling_tests.json` / `functions_definition.json`, followed by
  manual inspection of `function_calling_results.json` for valid JSON and
  correct schema.

## Example Usage

Given `data/input/functions_definition.json`:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
    "returns": {"type": "number"}
  }
]
```

and `data/input/function_calling_tests.json`:

```json
[{"prompt": "What is the sum of 40 and 2?"}]
```

Running:

```bash
uv run python -m src
```

produces `data/output/function_calling_results.json`:

```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40.0, "b": 2.0}
  }
]
```

## Resources

**References:**
- [A Guide to Structured Outputs Using Constrained Decoding — Aidan Cooper](https://www.aidancooper.co.uk/constrained-decoding/)
- [Awesome-LLM-Constrained-Decoding — curated paper list](https://github.com/Saibo-creator/Awesome-LLM-Constrained-Decoding)
- [Byte-Pair Encoding tokenization — Hugging Face LLM Course](https://huggingface.co/learn/llm-course/en/chapter6/5)
- [Tokenization algorithms — Hugging Face Transformers docs](https://huggingface.co/docs/transformers/tokenizer_summary#byte-pairencoding)
- [Function calling — OpenAI documentation](https://platform.openai.com/docs/guides/function-calling)
- [JSON Schema specification](https://json-schema.org/)
- [BPE tokenization overview — Hugging Face NLP course](https://huggingface.co/learn/nlp-course/chapter6/5)
- [JSON Schema specification](https://json-schema.org/)

**How AI was used:**
AI assistance was used for the readme