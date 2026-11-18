
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.pipeline import build_trip

# 1. Load the previously trained model
model_path = Path("models/pathsnap_model.joblib")
bundle = joblib.load(model_path)

model = bundle["model"]
features = bundle["features"]
k = bundle["candidate_k"]

print("\n====================================")
print("       PATHSNAP LIVE PREDICTION")
print("====================================")
print("Loaded model:", bundle["model_name"])
print("Candidate roads:", k)
print("Model loaded successfully!")

# 2. Select an unseen student-test trip
test_dir = Path("../data/Student folder/student_test")
trip_folders = sorted(p for p in test_dir.iterdir() if p.is_dir())

trip = trip_folders[0]

print("\nTesting trip:", trip.name)

# 3. Preprocess GPS and generate road candidates
df, stats = build_trip(trip, k=k)

if df.empty:
    raise RuntimeError("No candidate observations generated.")

print("GPS observations:", stats["observations"])
print("Candidate rows:", len(df))

# 4. Prepare the ten features
X = df[features].replace(
    [np.inf, -np.inf], np.nan
).fillna(0)

# 5. Use the trained model to predict probabilities
probabilities = model.predict_proba(X)[:, 1]

df["match_probability"] = probabilities

# 6. Display the three highest-scoring candidates
print("\n====================================")
print("          PREDICTIONS")
print("====================================")

for obs_id, group in df.groupby("obs_idx", sort=True):
    top = group.sort_values(
        "match_probability", ascending=False
    ).head(3)

    best = top.iloc[0]

    print(f"\nGPS Observation: {obs_id}")
    print(
        f"Selected road: Arc {int(best['arc_id'])}"
    )
    print(
        f"Match probability: "
        f"{best['match_probability'] * 100:.2f}%"
    )

    print("Top 3 candidates:")
    for _, candidate in top.iterrows():
        print(
            f"  Arc {int(candidate['arc_id'])}: "
            f"{candidate['match_probability'] * 100:.2f}%"
        )

    print(
        "Approximate reference match:",
        "YES" if best["y"] == 1 else "NO"
    )

    if obs_id >= 4:
        break

print("\n====================================")
print("LIVE PREDICTION COMPLETED")
print("====================================")
