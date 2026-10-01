Sim. Aqui está todo o conteúdo em um único ficheiro `README.md`:

````markdown name=README.md
*This project has been created as part of the 42 curriculum by gmateus-.*

# Call Me Maybe

## Description

Call Me Maybe is a Python project that turns natural-language requests into structured function calls for a small language model. Instead of asking the model to answer with plain text, the system selects the correct function from a known schema and extracts arguments in a format that can be executed by a machine.

The project targets the `Qwen/Qwen3-0.6B` model through the provided `llm_sdk`, and it uses constrained decoding techniques to keep outputs consistent with a known schema. The goal is to force the model to produce valid, machine-readable JSON rather than free-form text that may be ambiguous or invalid.

For example, given the prompt:

```text
What is the sum of 40 and 2?
```

the program should produce:

```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40.0,
    "b": 2.0
  }
}
```

This project addresses a common problem in small-model inference: a small language model may understand the user's request but still generate invalid JSON or unsupported values. Constrained decoding reduces this risk by restricting the possible next tokens according to the available functions and expected parameter types.

---

## Instructions

### Requirements

- Python 3.10 or newer
- `uv`
- The `llm_sdk` package provided with the project
- Internet access for the first model download

### Installation

Install the project dependencies with:

```bash
make install
```

The project uses `uv` to create and synchronize the Python environment.

The grading environment can also install the project with:

```bash
uv sync
```

### Running the project

The simplest way to run the project is:

```bash
make run
```

The equivalent direct command is:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

All command-line arguments are optional.

If no arguments are provided, the program uses these default paths:

```text
data/input/functions_definition.json
data/input/function_calling_tests.json
data/output/function_calling_results.json
```

### Custom paths

Custom input and output files can be provided:

```bash
uv run python -m src \
  --functions_definition path/to/functions_definition.json \
  --input path/to/function_calling_tests.json \
  --output path/to/function_calling_results.json
```

### Debugging

The program can be run under Python's debugger with:

```bash
make debug
```

### Linting and type checking

Run the required code-quality checks with:

```bash
make lint
```

This executes:

```bash
flake8 .
```

and:

```bash
mypy . \
  --warn-return-any \
  --warn-unused-ignores \
  --ignore-missing-imports \
  --disallow-untyped-defs \
  --check-untyped-defs
```

### Cleaning generated files

To remove Python caches and temporary files:

```bash
make clean
```

---

## Input Files

### Function definitions

The file `data/input/functions_definition.json` contains the functions available to the model.

Example:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  },
  {
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {
      "name": {
        "type": "string"
      }
    },
    "returns": {
      "type": "string"
    }
  }
]
```

Each function definition contains:

- `name`: the function name;
- `description`: an explanation of the function;
- `parameters`: the expected arguments and their types;
- `returns`: the return type.

### Prompt file

The file `data/input/function_calling_tests.json` contains the natural-language prompts to process.

Example:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?"
  },
  {
    "prompt": "Greet Alice"
  },
  {
    "prompt": "Reverse the string 'hello'"
  }
]
```

The input file must contain a JSON array. Each element must contain a non-empty `prompt` string.

---

## Output File

The program writes the results to:

```text
data/output/function_calling_results.json
```

Each result contains exactly:

- `prompt`;
- `name`;
- `parameters`.

Example:

```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 40.0,
      "b": 2.0
    }
  }
]
```

The output is serialized using Python's JSON module, with no additional explanatory text.

---

## Project Structure

```text
.
├── .flake8
├── .gitignore
├── Makefile
├── README.md
├── pyproject.toml
├── uv.lock
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── functions_definition.json
└── src/
    ├── __main__.py
    ├── app.py
    ├── decoder.py
    ├── io_utils.py
    └── models.py
```

The `llm_sdk` directory is provided as part of the 42 project infrastructure and is used by the application to interact with the language model.

### `src/__main__.py`

Provides the command-line entry point and parses the following arguments:

- `--functions_definition`;
- `--input`;
- `--output`.

### `src/app.py`

Orchestrates the complete application:

1. loads the function definitions;
2. loads the prompts;
3. initializes the model;
4. processes each prompt;
5. validates the result;
6. saves the output JSON file.

### `src/decoder.py`

Contains the constrained decoding implementation, function selection logic, and parameter extraction logic.

### `src/io_utils.py`

Contains helpers for:

- loading JSON files;
- validating input data;
- saving output data;
- handling filesystem and JSON errors.

### `src/models.py`

Contains the Pydantic models used to validate:

- function definitions;
- parameter definitions;
- return definitions;
- input prompts;
- output items.

---

## Algorithm Explanation

The project uses constrained decoding to restrict the model's possible outputs.

A language model normally generates one token at a time. At each generation step, it returns logits representing the score of every possible next token.

The project uses these logits as follows:

1. Encode the current prompt.
2. Ask the model for the logits of the next token.
3. Convert vocabulary entries into usable token text.
4. Reject tokens that would produce an invalid continuation.
5. Select the highest-scoring valid token.
6. Append the selected token to the input.
7. Repeat until the expected value is complete.

The main generation primitive is:

```python
ConstrainedDecoder.generate()
```

It receives two validation callbacks:

```python
is_valid_continuation
is_complete
```

`is_valid_continuation` determines whether a candidate token can still lead to a valid result.

`is_complete` determines whether the generated text is complete and can be returned.

### Function name selection

The available function names are used as the allowed candidate set.

For example:

```text
fn_add_numbers
fn_greet
fn_reverse_string
```

The decoder only allows tokens that are prefixes of at least one valid function name.

If the current generated text is:

```text
fn_
```

only tokens that can continue one of the available names are accepted.

This prevents the decoder from returning a function that does not exist in the function definition file.

The model's logits decide which valid function name is selected. The implementation does not use a simple keyword rule such as:

```python
if "greet" in prompt:
    ...
```

### Number parameters

Numbers are constrained using a partial-number grammar that accepts values such as:

```text
10
-5
3.14
-0.25
```

The decoder does not allow arbitrary text to be used as a numeric value.

When possible, numeric candidates are extracted from the user's prompt and the model selects among those candidates using constrained decoding.

### String parameters

Quoted strings are extracted from the prompt when possible.

For example:

```text
Reverse the string "hello world"
```

produces the candidate:

```text
hello world
```

The decoder can then select the appropriate candidate using the model's logits.

If no suitable quoted candidate exists, the implementation may use constrained free-form generation as a fallback.

### Boolean parameters

Boolean parameters are selected from the restricted set:

```text
true
false
```

The model is not allowed to produce arbitrary text for a boolean value.

---

## Design Decisions

### Pydantic validation

Pydantic is used for schema-related classes because the subject requires structured validation.

The following objects are validated:

- function definitions;
- parameter definitions;
- return definitions;
- input prompts;
- output objects.

This prevents malformed input data from silently entering the decoding pipeline.

### Candidate-based extraction

Small language models can struggle to reproduce exact values token by token. For example, the model may transform a simple string into a longer explanation or generate an invalid number.

To reduce this behavior, the implementation extracts possible values from the input prompt and uses constrained decoding to select among them.

This approach improves exact copying for:

- numbers;
- quoted strings;
- names;
- short text values.

### Model-driven function choice

The function name must be selected with the help of the LLM. The implementation therefore restricts the model's possible output to valid function names and lets the logits decide which candidate is preferred.

This satisfies the main requirement without using hardcoded function-selection keywords.

### Modular architecture

The code is split into small modules with separate responsibilities:

- command-line handling;
- application orchestration;
- model decoding;
- file input and output;
- schema validation.

This makes the project easier to test and maintain.

### Error handling

Input files are read using context managers and are validated before processing. Common errors such as missing files, invalid JSON, invalid schemas, and model initialization failures are caught and reported with clear messages.

---

## Performance Analysis

The project prioritizes reliable structured output over maximum generation speed.

### Main performance costs

The most expensive operations are:

- loading the Qwen model;
- loading the vocabulary;
- running inference for every decoding step;
- checking the possible vocabulary tokens;
- processing multiple prompts.

### Expected performance

The project is intended to process the provided evaluation dataset within the time limit specified by the subject. Actual execution time depends on:

- CPU or GPU availability;
- available memory;
- model download status;
- number of prompts;
- size of the vocabulary;
- number of generated tokens.

### Current benchmark status

Final accuracy and timing measurements should be recorded after running the complete dataset on the target evaluation machine.

The following values should be measured before submission:

- total number of prompts;
- number of correctly selected functions;
- number of correctly extracted parameters;
- total processing time;
- number of failed prompts;
- percentage of valid JSON outputs.

### Known bottleneck

The decoder may inspect many vocabulary tokens at each generation step. This provides strong control over valid output but increases computation time.

Potential future optimizations include:

- caching normalized vocabulary tokens;
- precomputing valid token groups;
- building a token-level trie;
- avoiding repeated vocabulary conversions;
- batching compatible operations.

---

## Challenges Faced

### Small language model limitations

The Qwen3-0.6B model is considerably smaller than most modern instruction-tuned models. It may produce explanations, repeated text, or malformed structures when unconstrained.

The decoding restrictions are used to make the output more predictable.

### Tokenization artifacts

Subword tokenizers may represent spaces and newlines with special markers such as:

```text
Ġ
▁
Ċ
ĉ
```

The decoder normalizes these markers before applying its validation rules.

### Exact value extraction

It is difficult for a small model to generate an exact string or number from scratch. A value may be repeated, extended, truncated, or followed by an explanation.

Candidate extraction combined with constrained selection reduces this problem.

### Ambiguous prompts

Some prompts contain multiple numbers or multiple possible strings. The decoder must select the values that best match the selected function while respecting the available schema.

### Dynamic function schemas

The function definitions are loaded at runtime and may change during evaluation. The implementation therefore cannot hardcode the provided example functions.

---

## Testing Strategy

The project should be tested at several levels.

### Input validation tests

Test the behavior for:

- missing files;
- invalid JSON;
- JSON objects where arrays are expected;
- empty function lists;
- empty prompts;
- missing required fields;
- unsupported parameter types;
- invalid function definitions.

### Decoder tests

Test:

- function selection from multiple candidates;
- function names with shared prefixes;
- negative numbers;
- decimal numbers;
- integer parameters;
- boolean parameters;
- quoted strings;
- empty strings;
- strings containing punctuation;
- repeated values;
- large values;
- prompts containing distractor numbers.

### Integration tests

Run the complete program using:

```bash
uv run python -m src
```

Then verify that:

- the process exits correctly;
- the output file is generated;
- the output is valid JSON;
- every result has the required keys;
- every selected function exists in the definitions;
- all required parameters are present;
- no unexpected parameters are present;
- parameter values have the expected types.

### Manual examples

Example prompts to test:

```text
What is the sum of 3 and 3?
```

```text
Greet Alice
```

```text
Reverse the string 'hello'
```

```text
Add -25 and -75.
```

```text
Please calculate 0.5 plus 0.25.
```

```text
Reverse the string "hello, world!"
```

---

## Example Usage

### Function definitions

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      }
    },
    "returns": {
      "type": "number"
    }
  }
]
```

### Prompts

```json
[
  {
    "prompt": "What is the sum of 40 and 2?"
  }
]
```

### Command

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Expected result

```json
[
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 40.0,
      "b": 2.0
    }
  }
]
```

---

## Resources

### Documentation and references

- [JSON Schema](https://json-schema.org/)
- [RFC 8259: The JSON Data Interchange Format](https://datatracker.ietf.org/doc/html/rfc8259)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Qwen model family](https://huggingface.co/Qwen)
- [Byte-Pair Encoding](https://huggingface.co/learn/llm-course/en/chapter6/5)
- [Function calling concepts](https://platform.openai.com/docs/guides/function-calling)
- [Constrained decoding overview](https://www.aidancooper.co.uk/constrained-decoding/)
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


## Notes

This project was developed for the 42 curriculum.

The main objective is to demonstrate how a small language model can be guided toward structured function calling using:

- constrained decoding;
- vocabulary-level token filtering;
- Pydantic validation;
- schema-driven parameter handling;
- robust JSON input and output management.
````