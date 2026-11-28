# QueryAlign RLVR-GRPO Dataset Registry, Checksums & Leakage Audit

## 1. Benchmark Datasets

### Spider 1.0 (Primary Dataset — Ingested)
- **Official Source**: Yale LILY Lab ([taoyds/spider](https://github.com/taoyds/spider), [EMNLP 2018](https://aclanthology.org/D18-1425/))
- **Configured Mirrors**:
  - Primary: `https://huggingface.co/datasets/GuokLIU/db-upload/resolve/main/spider.zip`
  - Fallback: `https://huggingface.co/datasets/zmmjjkk/SqurveBridge-benchmarks/resolve/main/spider.zip`
- **Local Archive (`data/raw/spider.zip`)**: `99,736,136` bytes | SHA-256: `5DDFF97BB1D421282C593E8D30CE0CE107270F4DD4A21D60EBA4BF287D5956B1`
- **License & Redistribution Notes**: Spider 1.0 is distributed under CC BY-SA 4.0 for academic/research use. Because individual upstream SQLite databases include third-party academic corpora (e.g., Restaurants, GeoQuery, Scholar, Academic, IMDB, Yelp — see [`data/raw/spider/README.txt`](file:///C:/Projects/queryalign-rlvr-grpo/data/raw/spider/README.txt)), raw databases remain gitignored and must not be redistributed without honoring upstream terms.

### BIRD Mini-Dev (Optional Extension — Not Ingested)
- **Source**: [bird-bench/mini_dev](https://github.com/bird-bench/mini_dev)
- **Status**: Optional/not ingested in the current training or evaluation manifests (`configs/data.yaml`: `status: "optional_not_ingested"`). If ingested later, pin a specific release/commit tag and select compatible SQLite `SELECT`-only tasks.

---

## 2. Processed Manifest Inventory & SHA-256 Checksums

| Split | File | Rows | Distinct DBs | Size (Bytes) | SHA-256 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Train** | [`data/processed/train_manifest.jsonl`](file:///C:/Projects/queryalign-rlvr-grpo/data/processed/train_manifest.jsonl) | 7,000 | 140 | 34,527,553 | `88FB8F1916DC0CDAAB2CB4BF97A57D09B09080E78BEDFEF43554CBBD1E0152E6` |
| **Dev (Held-out)** | [`data/processed/dev_manifest.jsonl`](file:///C:/Projects/queryalign-rlvr-grpo/data/processed/dev_manifest.jsonl) | 1,034 | 20 | 4,073,744 | `6496D3C410445D6EE0E2E2FF3104CFB80AC144D4A7237FCCC0D232C7FD2F581A` |
| **Data Zip Archive** | [`sql_grpo_data.zip`](file:///C:/Projects/queryalign-rlvr-grpo/sql_grpo_data.zip) | — | — | 98,055,129 | `2D04806399FA51B433B9322F42DB115CC15CB922993DAE27F9BDB31AECE650BB` |

---

## 3. Gold Execution QA & Leakage Audit Summary

Verified by `python -m sql_grpo.data.validate_dataset --config configs/data.yaml`:

- **Train Split (`7,000` rows, `train_spider.json` across `140` DBs)**:
  - `gold_status == "success"`: `6,743`
  - `gold_status == "empty"` (`gold_is_empty: true`): `254` (`3.63%`, retained with anti-hacking protection)
  - `gold_status == "error"`: `3` (`train_003153` on `assets_maintenance` referencing missing `Ref_Company_Types`; `train_004513` & `train_004514` on `document_management` using `ORDER BY ... LIMIT ... INTERSECT ...` disallowed by SQLite)
- **Dev Split (`1,034` rows, `dev.json` across `20` DBs)**:
  - `gold_status == "success"`: `983`
  - `gold_status == "empty"` (`gold_is_empty: true`): `49` (`4.74%`, retained in official evaluation denominator)
  - `gold_status == "error"`: `2` (`dev_000455` & `dev_000456` on `wta_1` exceeding the 5,000-row sandbox fetch limit on `players`; retained without rewriting official Spider labels)
- **Train/Dev Database Overlap**: **`0` shared `db_id` values** (`set()` — strict cross-database split).
- **Schema & AST Validation Across All `8,034` Rows**: `8,033 / 8,034` pass strict table+column AST validation (`1` upstream Spider defect: `train_003153` referencing missing `Ref_Company_Types`); `10` within-split identical `(db_id, question)` duplicates from upstream Spider; `6` generic question strings shared across disjoint databases (`0` same-DB duplicates across splits).

---

## 4. Pipeline Commands

```bash
# 1. Verify or acquire Spider raw archive with SHA-256 check & Zip-Slip safe extraction
python -m sql_grpo.data.download --config configs/data.yaml

# 2. Validate existing processed manifests, read-only DB readability, ASTs, and leakage
python -m sql_grpo.data.validate_dataset --config configs/data.yaml

# 3. Validate preprocessing pipeline against configs/data.yaml (audit-only mode)
python -m sql_grpo.data.preprocess --config configs/data.yaml --validate-only
```

---

## 5. Academic Citations

```bibtex
@inproceedings{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and Radev, Dragomir},
  booktitle={EMNLP},
  year={2018}
}
```
