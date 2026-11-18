
import json
import heapq
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline import (
    DATA, FEATURES, build_trip, read_trip, project
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ROOT / "models"

K = 8
MAX_HOPS = 5
SIGMA_M = 60.0
UNREACHABLE_PENALTY = 8.0


def load_model():
    path = MODELS / "tuned_model.joblib"
    if path.exists():
        print("Loading tuned HistGradientBoosting model")
        return joblib.load(path)

    bundle = joblib.load(MODELS / "pathsnap_model.joblib")
    return bundle["model"] if isinstance(bundle, dict) else bundle


def make_graph(arcs, nodes):
    uv = arcs[["u", "v"]].to_numpy()
    valid = (
        (uv[:, 0] >= 0) & (uv[:, 1] >= 0)
        & (uv[:, 0] < len(nodes))
        & (uv[:, 1] < len(nodes))
    )
    arc_ids = np.where(valid)[0]
    uv = uv[valid]

    adjacency = {}
    for arc_id, (u, v) in zip(arc_ids, uv):
        length = float(np.linalg.norm(nodes[v] - nodes[u]))
        adjacency.setdefault(int(u), []).append((int(v), length))

    return arc_ids, uv, adjacency


def shortest_path_bounded(adjacency, source, target, max_hops=MAX_HOPS):
    if source == target:
        return 0.0

    queue = [(0.0, source, 0)]
    best = {(source, 0): 0.0}

    while queue:
        distance, node, hops = heapq.heappop(queue)

        if node == target:
            return distance

        if hops >= max_hops:
            continue

        if distance > best.get((node, hops), float("inf")):
            continue

        for nxt, edge_length in adjacency.get(node, []):
            new_distance = distance + edge_length
            key = (nxt, hops + 1)

            if new_distance < best.get(key, float("inf")):
                best[key] = new_distance
                heapq.heappush(
                    queue, (new_distance, nxt, hops + 1)
                )

    return None


def decode_trip(folder, model):
    trip_id = folder.name
    print("Processing trip:", trip_id)

    df, _ = build_trip(folder, K)
    if df.empty:
        return None

    track, raw_nodes, arcs, route = read_trip(folder)

    track = (
        track.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .drop_duplicates("ts")
        .sort_values("ts")
        .iloc[::3]
        .reset_index(drop=True)
    )

    lon0 = float(track.lon.mean())
    lat0 = float(track.lat.mean())

    P = project(track.lon, track.lat, lon0, lat0)
    N = project(raw_nodes.lon, raw_nodes.lat, lon0, lat0)

    valid_ids, uv, adjacency = make_graph(arcs, N)

    arc_geometry = {}
    for arc_id, (u, v) in zip(valid_ids, uv):
        A, B = N[u], N[v]
        vector = B - A
        length = float(np.linalg.norm(vector))
        arc_geometry[int(arc_id)] = (int(u), int(v), A, B, length)

    X = (
        df[FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    probabilities = model.predict_proba(X)[:, 1]
    df = df.copy()
    df["probability"] = probabilities

    observations = []
    for obs_idx, group in df.groupby("obs_idx", sort=True):
        group = group.copy()
        arcs_here = group.arc_id.to_numpy(dtype=int)
        probs = group.probability.to_numpy(dtype=float)

        # Normalised candidate emission probabilities
        probs = np.clip(probs, 1e-9, 1.0)
        probs = probs / probs.sum()

        observations.append({
            "obs_idx": int(obs_idx),
            "arcs": arcs_here,
            "emission": np.log(probs),
            "labels": group.y.to_numpy(dtype=int)
        })

    if len(observations) < 2:
        return None

    # Project each GPS observation onto each candidate segment.
    fractions = []
    for obs in observations:
        row = []
        point = P[obs["obs_idx"]]

        for arc_id in obs["arcs"]:
            u, v, A, B, length = arc_geometry[int(arc_id)]
            vector = B - A
            denom = max(float(np.dot(vector, vector)), 1e-9)
            fraction = np.clip(
                np.dot(point - A, vector) / denom, 0.0, 1.0
            )
            row.append(float(fraction))

        fractions.append(row)

    # Classifier-only prediction
    baseline_indices = [
        int(np.argmax(obs["emission"]))
        for obs in observations
    ]

    # Viterbi initialisation
    scores = observations[0]["emission"].copy()
    backpointers = []

    for t in range(1, len(observations)):
        prev = observations[t - 1]
        curr = observations[t]

        gps_distance = float(
            np.linalg.norm(
                P[curr["obs_idx"]] - P[prev["obs_idx"]]
            )
        )

        new_scores = np.full(
            len(curr["arcs"]), -np.inf
        )
        pointers = np.zeros(
            len(curr["arcs"]), dtype=int
        )

        for j, curr_arc in enumerate(curr["arcs"]):
            curr_u, curr_v, _, _, curr_len = (
                arc_geometry[int(curr_arc)]
            )

            for i, prev_arc in enumerate(prev["arcs"]):
                prev_u, prev_v, _, _, prev_len = (
                    arc_geometry[int(prev_arc)]
                )

                if int(prev_arc) == int(curr_arc):
                    network_distance = (
                        fractions[t][j] - fractions[t - 1][i]
                    ) * prev_len

                    if network_distance < 0:
                        network_distance = (
                            prev_len * (1 - fractions[t - 1][i])
                            + curr_len * fractions[t][j]
                        )
                else:
                    middle = shortest_path_bounded(
                        adjacency, prev_v, curr_u
                    )

                    if middle is None:
                        network_distance = None
                    else:
                        network_distance = (
                            (1 - fractions[t - 1][i]) * prev_len
                            + middle
                            + fractions[t][j] * curr_len
                        )

                if network_distance is None:
                    transition = -UNREACHABLE_PENALTY
                else:
                    transition = (
                        -abs(network_distance - gps_distance)
                        / SIGMA_M
                    )

                candidate_score = (
                    scores[i] + transition
                )

                if candidate_score > new_scores[j]:
                    new_scores[j] = candidate_score
                    pointers[j] = i

        scores = new_scores + curr["emission"]
        backpointers.append(pointers)

    # Backtrack optimal state sequence
    state = int(np.argmax(scores))
    decoded_indices = [state]

    for pointers in reversed(backpointers):
        state = int(pointers[state])
        decoded_indices.append(state)

    decoded_indices.reverse()

    baseline_arcs = [
        int(obs["arcs"][idx])
        for obs, idx in zip(observations, baseline_indices)
    ]

    decoded_arcs = [
        int(obs["arcs"][idx])
        for obs, idx in zip(observations, decoded_indices)
    ]

    baseline_correct = [
        int(obs["labels"][idx])
        for obs, idx in zip(observations, baseline_indices)
    ]

    decoded_correct = [
        int(obs["labels"][idx])
        for obs, idx in zip(observations, decoded_indices)
    ]

    def continuity(route_arcs):
        if len(route_arcs) < 2:
            return 0.0

        connected = 0
        for a, b in zip(route_arcs[:-1], route_arcs[1:]):
            if a == b:
                connected += 1
                continue

            _, v, _, _, _ = arc_geometry[a]
            u2, _, _, _, _ = arc_geometry[b]

            if shortest_path_bounded(adjacency, v, u2) is not None:
                connected += 1

        return connected / (len(route_arcs) - 1)

    result = {
        "trip_id": trip_id,
        "observations": len(observations),
        "classifier_top1_accuracy": float(np.mean(baseline_correct)),
        "hmm_top1_accuracy": float(np.mean(decoded_correct)),
        "classifier_sequence_accuracy": float(all(baseline_correct)),
        "hmm_sequence_accuracy": float(all(decoded_correct)),
        "classifier_route_continuity": continuity(baseline_arcs),
        "hmm_route_continuity": continuity(decoded_arcs)
    }

    # Predicted route output, not ground truth
    route_output = pd.DataFrame({
        "obs_idx": [o["obs_idx"] for o in observations],
        "classifier_arc_id": baseline_arcs,
        "hmm_arc_id": decoded_arcs,
        "latitude": [track.iloc[i].lat for i in range(len(observations))],
        "longitude": [track.iloc[i].lon for i in range(len(observations))]
    })

    route_output.to_csv(
        RESULTS / f"hmm_route_{trip_id}.csv", index=False
    )

    return result


def main():
    model = load_model()
    test_folder = DATA / "student_test"

    results = []
    for folder in sorted(test_folder.iterdir()):
        if not folder.is_dir():
            continue

        try:
            result = decode_trip(folder, model)
            if result:
                results.append(result)
        except Exception as exc:
            print("ERROR", folder.name, str(exc))

    if not results:
        raise RuntimeError("No student-test trips were processed.")

    detail = pd.DataFrame(results)
    detail.to_csv(
        RESULTS / "hmm_trip_comparison.csv", index=False
    )

    summary = {}
    for column in detail.columns:
        if column == "trip_id":
            continue
        summary[column] = float(detail[column].mean())

    summary["decoder"] = "HMM + Viterbi"
    summary["model"] = "HistGradientBoosting"

    with open(RESULTS / "hmm_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nHMM COMPARISON")
    print(detail.to_string(index=False))
    print("\nAVERAGES")
    print(json.dumps(summary, indent=2))
    print("\nSaved results in:", RESULTS)


if __name__ == "__main__":
    main()
