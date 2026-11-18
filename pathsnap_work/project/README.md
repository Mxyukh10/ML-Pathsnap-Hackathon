# PathSnap working baseline

## Dataset layout
Expected extracted layout: `../data/Student folder/{train,student_test}/<trip_id>/<trip_id>.<track|nodes|arcs|route>`.

## Run
From this project directory:

```bash
python -m pip install -r requirements.txt
python src/pipeline.py --quick
```

Outputs: `models/pathsnap_model.joblib`, `results/model_comparison.csv`, `results/train_trip_eda.csv`, `results/student_test_metrics.json`, and `results/run_summary.json`.

## Important methodology note
The supplied `.route` contains a chronological sequence of arc IDs but no explicit per-GPS-observation labels. This baseline creates approximate observation labels by nearest spatial road-segment alignment to the supplied route. Candidate generation uses the nearest road-segment midpoints; candidate miss rate is recorded. This is a transparent baseline assumption, not verified ground truth. Validate it and explain it during evaluation. The evaluator partition is not loaded by this script.
