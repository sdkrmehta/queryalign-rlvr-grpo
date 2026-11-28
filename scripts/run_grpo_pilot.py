"""CLI script: Run GRPO pilot configuration via sql_grpo.training.train_grpo."""
import sys
from sql_grpo.training.train_grpo import main

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/grpo_pilot.yaml"])
    main()
