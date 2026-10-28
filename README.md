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

The vocabulary file returned by the SDK is read as a `{token: id}` JSON map. Space and newline markers of the tokenizer (`Ġ`, `▁`, `Ċ`, `ĉ`) are converted back to real characters before any rule is applied.

### Pipeline for one prompt

1. The function name is chosen first (see below).
2. The parameters of the chosen function are then filled one by one, in the order they appear in the function definition. How each value is produced depends on its type.
3. The result is returned as `{"name": ..., "parameters": {...}}`.

Most values are produced by **closed-set selection**: the code lists the exact strings that are acceptable, and the model picks one of them through the same token-by-token mechanism used for function names. A value that does not exist in the prompt (for example a regular expression) is **generated** by the model instead.

### Function name selection

The available function names are used as the allowed candidate set.

For example:

```text
fn_add_numbers
fn_greet
fn_reverse_string
```

The decoder only allows tokens that are prefixes of at least one valid function name. Generation is complete when the text equals one of the names.

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

The text the model reads before choosing contains the list of functions and, for each function, one line that pairs its description with its name. These lines are built at runtime from the definitions file, so nothing is hardcoded. Without them, short imperative requests such as `Say hello to Maria` were sent to the wrong function.

### Number parameters

Every numeric substring of the prompt (optional minus sign, digits, optional decimal part) becomes a candidate. Repeated numbers are kept, so `0 and 0` offers two candidates.

For each number parameter the model selects one of the remaining candidates with the same prefix-based decoding, and that occurrence is removed from the list.

After all number parameters are filled, the chosen values are sorted by the position where they appear in the prompt and assigned to the parameters in definition order (first number written, first parameter). The model decides **which** numbers are used, which matters when the prompt contains numbers that are not arguments (for example `A lorena tem 50 anos e quer saber quanto e 70 mais 47`); the code decides their **order**.

If a number parameter has no candidate left, a fallback generation accepts only text that matches a partial number (`10`, `-5`, `3.14`) and stops after a complete number followed by whitespace, with a small token limit.

### String parameters

Quoted text (single or double quotes) is extracted from the prompt. An empty pair of quotes gives an empty-string candidate. If the prompt has no quoted text, the distinct alphabetic words of the prompt are the candidates. Candidates already used by another string parameter are removed.

For example:

```text
Reverse the string "hello world"
```

produces the candidate:

```text
hello world
```

When candidates remain, the model selects one with the prefix-based decoding. The text it reads is a partial JSON call: the function signature and description, the request, and the call written up to the opening quote of the value, with the parameters already decided written before it.

When no candidate remains, the value does not exist in the prompt and has to be written by the model. This is what happens for the regular expression of a request such as `Replace all numbers in "..." with NUMBERS`. In this case:

- the text read by the model ends right after the opening quote of the value and also contains two solved examples (one where the replacement is copied from the request, one where the request names a symbol and the value is the symbol itself);
- at every step, tokens that cannot appear raw inside a JSON string (line breaks, tabs, special tokens) are masked, and the highest-scoring remaining token is taken;
- if that token contains an unescaped double quote, the value ends there and only the text before the quote is kept (maximum 48 tokens);
- the body is decoded as a JSON string, so the model writes `\\d+` and the stored value is `\d+`;
- a single leading space produced by the tokenizer is removed.

### Boolean parameters

Boolean parameters are selected from the restricted set:

```text
true
false
```

The model is not allowed to produce arbitrary text for a boolean value.

### What the model decides and what the code decides

| Decision | Made by |
|---|---|
| Function name | Model (logits over the valid names) |
| Which numbers are arguments, when the prompt has extra numbers | Model |
| Which quoted text or word goes to which string parameter, when there are several | Model |
| Boolean value | Model |
| Regular expression, replacement, or any string that is not in the prompt | Model (generated token by token) |
| Order of the numbers among the parameters | Code (position in the prompt) |
| Value when only one candidate exists | Code (no model call is made) |

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

### Closed-set selection for values that exist in the prompt

Early versions let the model generate every value token by token, only checking that the shape stayed valid. With a 0.6B model this was unreliable: a number kept growing (`2` became `2e+22`), a name was followed by the model's own commentary, and a word was repeated without end.

Tuning the stopping rule did not fix it, so the approach changed: the exact acceptable strings are listed from the prompt and the model only chooses among them. A selected value is always an exact copy of something in the prompt, so it cannot be truncated, extended or repeated.

### Generation only for values that are not in the prompt

Some arguments cannot be copied. For these, the value is generated as the body of a JSON string. The text read by the model is a JSON call that stops at the opening quote of the value, which is the continuation a language model finds natural, and the end of the value is the closing quote.

The two solved examples in that text are the only fixed text about a kind of function in the project. They are used only when a value has to be invented, so they do not influence names, numbers or quoted strings.

### Positional order for numbers

Choosing a number independently for each parameter produced swapped values (`-10 and 4` gave `a=4, b=-10`). Sorting the chosen numbers by their position in the prompt keeps the order of the sentence.

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

### How accuracy was measured

There is no reference answer file. Each result was checked by hand against what the prompt asks for. The output file is always valid JSON because it is written with Python's JSON module, and parameter types follow the definition by construction (number gives a float, string gives a string, boolean gives a boolean).

### Results

| Prompt set | Prompts | Correct function | Correct parameters | Time |
|---|---|---|---|---|
| Subject-style set (add, greet, reverse) | 51 | TBD | TBD | TBD |
| Extended set (adds square root and regex substitution) | 11 | 11 | 11 | TBD |

Time was measured on: TBD (CPU or GPU, memory).

In the extended set, the regular expression generated for `Replace all vowels in 'Programming is fun' with asterisks` is `([aeiou])`. It is correct for that text, but it does not match uppercase vowels (see Known Limitations).

### Main performance costs

The most expensive operations are:

- loading the Qwen model;
- reading the vocabulary file;
- running inference for every decoding step;
- scoring the whole vocabulary at every decoding step.

### Known bottleneck

At each generation step the decoder converts and checks every token of the vocabulary in Python. This gives strong control over valid output but increases computation time. The vocabulary file is also read again for every prompt.

Possible optimizations:

- reading the vocabulary and converting token texts once for the whole run;
- precomputing the tokens that can start each candidate;
- building a token-level trie;
- skipping the full-vocabulary scan when only one candidate is possible.

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

At first only the space markers were converted. A newline token (`Ċ`) was therefore not recognised as a line break, and the model's commentary leaked into a string value (`shrek` followed by a whole sentence). Converting the newline and tab markers fixed it.

The tokenizer also attaches a space to some tokens: the model chose a token for ` *` where `*` was wanted, and the value started with a space. The value is now stripped of that single leading space after the model has chosen its tokens.

### Exact value extraction

It is difficult for a small model to generate an exact string or number from scratch. Values were repeated, extended, truncated, or followed by an explanation. One stopping rule treated a single digit as a complete number and cut `265` to `2`.

The fix was closed-set selection (see Design Decisions), not a better stopping rule.

### Choosing the wrong function

Requests that start with a verb phrase, such as `Say hello to Maria` or `Say good morning to Maria`, were sent to `fn_add_numbers`, and the same prompt gave the same wrong answer every time because decoding is greedy. Adding the description-to-name lines to the text read by the model fixed every phrasing that was tested.

### Ambiguous prompts

Some prompts contain multiple numbers or multiple possible strings. The decoder must select the values that best match the selected function while respecting the available schema.

Specific cases that had to be handled:

- the same number twice (`0 and 0`): repeated numbers are kept, and one occurrence is removed per use;
- an empty string (`Reverse the string ''`): the quote pattern accepts an empty pair, otherwise the value silently became another word of the prompt;
- numbers in the wrong order: see Positional order for numbers;
- a number that is not an argument (an age mentioned next to the real sum): the model selects the numbers.

### Values that are not in the prompt

The regular expression of a substitution request cannot be copied. The first attempt used the same generation as for other strings and produced values such as `34,233,`. Three changes were needed: the text read by the model became a JSON call ending inside the opening quote, characters such as `[`, `]` and `\` were allowed, and generation stopped at the closing quote instead of at the first whitespace.

Two further problems appeared after that. The model copied `34` from the source text as the regex, which a solved example fixed. Then the model wrote the word `asterisk` instead of `*`, which a second example fixed. Because these examples were added after observing failures, other phrasings may still fail.

### Dynamic function schemas

The function definitions are loaded at runtime and may change during evaluation. The implementation therefore cannot hardcode the provided example functions.

---

## Known Limitations

- **Frequent words win over names.** In `Tell Maria hello` the name selected is `hello`, not `Maria`. The same prompt gives the same result every time.
- **Multi-word names.** When a name is not quoted, only one word is selected (`John Doe` gives `John`).
- **Numbers.** Numbers are read as digits with an optional sign and decimal part. Thousands separators (`1,000`) and ranges (`3-4`) are read wrongly, and numbers written in words are not extracted.
- **Generated values are not verified.** A generated regex may be valid but too narrow (`([aeiou])` misses uppercase vowels), and the result depends on how the request is phrased.
- **Quotes inside sentences.** Any pair of single or double quotes is treated as quoted text, so two apostrophes in the same sentence (`it's Bob's party`) can be read as a quoted string.
- **Non-ASCII text.** Only the space and newline markers are converted back to text. Byte-level tokens for accented characters are not, so values with accented letters may be selected or generated wrongly.
- **Parameter types.** Only string, number, integer and boolean parameters are supported. Other types raise an error.

---

## Testing Strategy

The project was tested by running the complete program with `make run` on several prompt sets and reading every result by hand.

### Prompt sets

- **51 prompts with the three subject functions** (add, greet, reverse):
  - sums with negative, decimal, very large and repeated numbers (`0 and 0`, `3 and 3`);
  - a sum with an extra number that is not an argument;
  - requests in English and Portuguese;
  - greetings in lowercase and uppercase, with titles (`Dr. Smith`) and with two-word names;
  - strings with spaces, punctuation, special characters and an empty string.
- **11 prompts with extra functions**: square root, and substitution with a regular expression (quoted text with an apostrophe inside, a request with three quoted texts).
- **20 greeting phrasings with the same name**, used to find which wording sent the request to the wrong function. A prompt repeated three times gave the same result each time, which confirms that decoding is deterministic.

### Method

Each wrong result was turned into a test prompt, the cause was found in the code, and the full sets were run again after every change to check that nothing that worked before had broken.

### Static checks

`make lint` (flake8 and mypy with the flags required by the subject): result TBD.

### Not done

The checks were manual. There is no automated test suite in the repository.

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

### Example with a generated value

For a prompt such as:

```text
Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS
```

the result was:

```json
{
  "name": "fn_substitute_string_with_regex",
  "parameters": {
    "source_string": "Hello 34 I'm 233 years old",
    "regex": "\\d+",
    "replacement": "NUMBERS"
  }
}
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