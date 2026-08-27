import argparse
from .app import run_app


def main() -> None:
    """Program entry point."""
    parser = argparse.ArgumentParser(
        description="Function calling tool powered by a small LLM."
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        help="Path to the function definitions JSON file.",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        help="Path to the input prompts JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file.",
    )
    args = parser.parse_args()

    code = run_app(
        functions_path=args.functions_definition,
        input_path=args.input,
        output_path=args.output,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
