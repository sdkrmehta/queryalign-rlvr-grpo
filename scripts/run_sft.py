"""CLI script: Run QLoRA SFT training via sql_grpo.training.train_sft."""
import sys
from sql_grpo.training.train_sft import main

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/sft.yaml"])
    main()
