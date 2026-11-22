"""CLI interface for SQL-GRPO."""

import argparse
import sys
from sql_grpo import __version__


def doctor():
    """Validates the runtime environment, SQLite engine, and core dependencies."""
    print(f"=== SQL-GRPO Environment Doctor (v{__version__}) ===")
    import sqlite3
    import platform
    print(f"Python Version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"SQLite Version: {sqlite3.sqlite_version}")

    try:
        import torch
        print(f"PyTorch Version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch: Not installed")

    try:
        import sqlglot
        print(f"SQLGlot Version: {sqlglot.__version__}")
    except ImportError:
        print("SQLGlot: Not installed")

    try:
        import transformers
        print(f"Transformers Version: {transformers.__version__}")
    except ImportError:
        print("Transformers: Not installed")

    try:
        import trl
        print(f"TRL Version: {trl.__version__}")
    except ImportError:
        print("TRL: Not installed")

    print("=====================================================")


def main():
    parser = argparse.ArgumentParser(description="SQL-GRPO Command Line Tool")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check runtime and environment compatibility")

    args = parser.parse_args()
    if args.subcommand == "doctor":
        doctor()


if __name__ == "__main__":
    main()
