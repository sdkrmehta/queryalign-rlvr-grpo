"""CLI script: Validate/prepare dataset manifests via sql_grpo.data.preprocess."""
import sys
from sql_grpo.data.preprocess import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--config", "configs/data.yaml", "--validate-only"])
    main()
