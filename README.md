# NERvis — Fault Analysis Tool and Evaluation Framework

This repository contains the software artifact that accompanies the paper *When "John Doe" Becomes "Jon Dough": An Industrial Study on Automated Natural Language Anonymization*. It provides two components:

- `nervis.py` — the interactive fault-analysis tool introduced in Section *Fault Analysis* of the paper.
- `evaluation_framework.ipynb` — the evaluation pipeline used to compute span-level performance metrics and per-entity indices.

## Mock example

The original data ingestion has been removed because the underlying customer-service transcripts are proprietary and cannot be released. Both scripts now ship with a small in-memory mock dataset so they can be executed standalone for demonstration and reproducibility. The mock dataset reproduces the example shown in the paper's fault-analysis figure and uses the PII label schema defined in the paper (*Table: Category groups and labels*).

All identifiers in the mock data (names, streets, postal codes) are randomly chosen and do not correspond to real individuals or addresses. The example is designed to illustrate the two evaluation policies defined in the paper:

- **Card 1** shows a prediction whose span boundaries match the ground truth but whose label differs (`NAME_GIVEN` instead of `NAME`). The prediction is correct under the Exact policy and incorrect under the Strict policy.
- **Card 2** shows a prediction whose span boundaries and label both match the ground truth (`LOCATION_ZIP`). The prediction is correct under both policies.

## Requirements

The scripts were developed and tested with Python 3.11. Install the dependencies with:

```bash
pip install dash dash-bootstrap-components pandas plotly nervaluate jupyter
```

## Running the fault-analysis tool

From the repository root, run:

```bash
python nervis.py
```

The Dash application starts on `http://127.0.0.1:8050`. Open this address in a browser to interact with the tool. The interface displays the selected policy and measure at the top, followed by each predicted span in its textual context alongside the corresponding ground-truth annotations.

## Running the evaluation notebook

Launch Jupyter and open the notebook:

```bash
jupyter notebook evaluation_framework.ipynb
```

Execute the cells in order. The notebook constructs the mock dataset, runs the `nervaluate` evaluator across the four evaluation policies (`strict`, `exact`, `partial`, `ent_type`), and produces the two final dataframes:

- `privateai_evaluation_metrics_df` — overall and per-entity metrics (precision, recall, F1, counts).
- `privateai_indices_df` — per-entity index buckets (correct, incorrect, partial, missed, spurious) for each policy.

## Notes

Both scripts are intended as a reference implementation of the evaluation procedure and fault-analysis workflow described in the paper. To apply them to a different corpus, replace the mock `MOCK_TEXT`, `pred_labels` and `true_labels` objects in `nervis.py` and `evaluation_framework.ipynb` with data from the target dataset in the same schema.
