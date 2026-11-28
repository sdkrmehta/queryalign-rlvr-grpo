"""CLI script: Evaluate untuned base model Zero-Shot baseline."""
import sys
from sql_grpo.evaluation.run_benchmark import main

if __name__ == "__main__":
    if "--model" not in sys.argv:
        sys.argv.extend(["--model", "base"])
    main()
