from src.app import run_app
import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
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
