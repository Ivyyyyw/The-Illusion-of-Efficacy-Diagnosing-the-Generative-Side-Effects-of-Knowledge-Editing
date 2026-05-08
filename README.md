# MyEasyEdit Anonymous

This repository contains the anonymized code and data for evaluating knowledge editing methods under free-form, multi-question generation. The main experiment compares `prompt_v2`, `ROME`, `FT-M`, `GRACE`, and `WISE` on a unified 51-instance evaluation set built for the paper.

Contact information is withheld during anonymous review.

## Repository Structure

```text
MyEasyEdit_anonymous/
├── run_merged_eval.sh
│   └── Main entry point for regenerating merged-evaluation outputs.
├── code/
│   ├── intervention/
│   │   ├── run_with_summary_metrics_no_acc.py
│   │   │   └── Runs each editing method and records free-form responses.
│   │   ├── eval_hallu.py
│   │   │   └── Evaluation helpers from the hallucination-editing workflow.
│   │   └── util.py
│   │       └── Shared utilities.
│   ├── easyeditor/
│   │   └── Local EasyEdit implementation used by the runner.
│   ├── hparams/
│   │   └── Method/model hyperparameter files.
│   └── hallucination_editor.py
│       └── Editing wrapper utilities.
├── data/
│   ├── merged_eval_dataset.json
│   │   └── Main 51-instance evaluation dataset used by the paper.
│   ├── ft_train_test.json
│   │   └── FT-M training/evaluation split consumed by the main runner.
│   ├── counterfact_v1.json
│   ├── test_with_hops.json
│   ├── test.csv
│   ├── FT_data.jsonl
│   └── add_hop_questions.py
│       └── Legacy source data and construction helpers kept for traceability.
├── results/
│   ├── merged_eval_dataset_prompt_v2_post.json
│   ├── merged_eval_dataset_ROME_post.json
│   ├── merged_eval_dataset_FT-M_post.json
│   ├── merged_eval_dataset_GRACE_post.json
│   ├── merged_eval_dataset_WISE_post.json
│   │   └── Released labeled outputs used for analysis.
│   ├── recompute_edit_success.py
│   └── review_eval_labels.py
│       └── Post-processing and label review utilities.
├── requirements.txt
└── README.md
```

Generated outputs from fresh runs are written under `results/merged_eval/` and are ignored by git. The flat JSON files under `results/` are the released labeled outputs.

## Installation

```bash
conda create -n easyedit python=3.9
conda activate easyedit
pip install -r requirements.txt
```

The experiments use `meta-llama/Meta-Llama-3-8B-Instruct`. Make sure your environment has access to the gated model, for example by setting a Hugging Face token:

```bash
export HF_TOKEN="your_huggingface_token"
```

## Main Evaluation

Run the full merged-evaluation pipeline:

```bash
bash run_merged_eval.sh
```

Run a subset of methods or samples:

```bash
bash run_merged_eval.sh ROME 0
bash run_merged_eval.sh "prompt_v2 ROME" 0 3
bash run_merged_eval.sh FT-M 0 51 21
```

Arguments:

```text
bash run_merged_eval.sh [methods] [gpu_id] [data_size] [start_index]
```

By default, the script runs:

```text
prompt_v2 ROME FT-M GRACE WISE
```

The runner reads `data/merged_eval_dataset.json`, uses method hyperparameters from `code/hparams/`, and writes generated outputs to:

```text
results/merged_eval/<METHOD>/llama3_8b/merged_eval_dataset_<METHOD>_post.json
```

## Released Results

The released labeled outputs are:

```text
results/merged_eval_dataset_prompt_v2_post.json
results/merged_eval_dataset_ROME_post.json
results/merged_eval_dataset_FT-M_post.json
results/merged_eval_dataset_GRACE_post.json
results/merged_eval_dataset_WISE_post.json
```

Each result file contains the main response, evaluation-question responses, multi-turn responses, and labels such as `edit_success`, `issue_type`, and `revert_turn`.

## Post-Processing Utilities

The scripts in `results/` support label checking and recomputation:

```bash
python results/recompute_edit_success.py
python results/review_eval_labels.py
```

Add `--apply` to scripts that support it when you want to write updates.

## Notes

- Model weights and generated fresh-run outputs are not committed.
- The `data/merged_eval_dataset.json` file is the canonical evaluation dataset for the paper.
- Legacy source data files are retained only to make the dataset construction traceable.
- This anonymized release intentionally excludes paper source files and large intermediate artifacts.

## License

MIT License
