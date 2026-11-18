
import time
import joblib
import pandas as pd

from pathlib import Path

from src.pipeline import DATA
from src.hmm_decoder import decode_trip

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "tuned_model.joblib"
RESULTS = ROOT / "results"

_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Tuned model not found: {MODEL_PATH}"
            )
        _model = joblib.load(MODEL_PATH)

    return _model


def get_student_test_folder():
    candidates = [
        DATA / "student_test",
        DATA / "student-test",
        DATA / "Student_test"
    ]

    for folder in candidates:
        if folder.is_dir():
            return folder

    raise FileNotFoundError(
        "Student test partition folder was not found."
    )


def list_trips():
    folder = get_student_test_folder()

    return sorted(
        p.name for p in folder.iterdir()
        if p.is_dir()
    )


def predict_trip(trip_id: str):
    start = time.perf_counter()

    folder = get_student_test_folder()
    allowed = {p.name for p in folder.iterdir() if p.is_dir()}

    if trip_id not in allowed:
        raise ValueError("Unknown or unavailable trip ID.")

    model = get_model()

    # Runs the existing classifier + HMM/Viterbi implementation.
    metrics = decode_trip(folder / trip_id, model)

    if metrics is None:
        raise ValueError("Could not process this trip.")

    # The existing decoder writes its predicted sequence to this CSV.
    route_file = RESULTS / f"hmm_route_{trip_id}.csv"

    if not route_file.exists():
        raise RuntimeError("HMM route output was not generated.")

    route_df = pd.read_csv(route_file)

    # These are GPS observation coordinates in [latitude, longitude]
    # order, as currently written by hmm_decoder.py.
    coordinates = route_df[
        ["latitude", "longitude"]
    ].astype(float).values.tolist()

    arc_ids = route_df["hmm_arc_id"].astype(int).tolist()

    elapsed = time.perf_counter() - start

    return {
        "trip_id": trip_id,
        "status": "success",
        "route_coordinates": coordinates,
        "predicted_arc_ids": arc_ids,
        "metrics": metrics,
        "inference_time_seconds": round(elapsed, 4)
    }
