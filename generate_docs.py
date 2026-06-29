from pathlib import Path


README_PATH = Path("README.md")


def main() -> None:
    if not README_PATH.exists():
        raise FileNotFoundError("README.md is missing; restore the canonical project README.")

    print("Documentation is consolidated in README.md.")
    print("No additional Markdown files are generated.")


if __name__ == "__main__":
    main()
