"""CLI script: Run full RLVR-GRPO training via sql_grpo.training.train_grpo."""
import sys
from sql_grpo.training.train_grpo import main

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/grpo.yaml"])
    main()
