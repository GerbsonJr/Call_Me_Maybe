*This project has been created as part of the 42 curriculum by GerbsonJr.*

# Call_Me_Maybe

## Description

Call_Me_Maybe is a Python project that converts natural-language prompts into structured function calls.

Given an input like:

- `What is the sum of 2 and 3?`
- `Greet shrek`
- `Reverse the string 'hello'`

the program selects the most appropriate function from `functions_definition.json` and extracts the required parameters into a strict JSON output format.

The goal of this project is to demonstrate function calling with an LLM while keeping the output valid, structured, and machine-readable.

## Instructions

### Requirements
- Python 3.10+
- `uv`

### Install dependencies
```bash
uv sync