# Preprocessing and feature engineering run summary

- Training trips inspected: **80**
- GPS observations after dropping invalid rows, duplicate timestamps, sorting, and taking every third observation: **69,495**
- Road nodes across trips (sum): **1,933,485**
- Road arcs across trips (sum): **3,876,904**
- Mean candidate miss rate per trip: **10.7%**
- Overall candidate miss rate weighted by observations: **13.7%**
- Candidate generation: nearest 8 road-edge midpoints, followed by exact point-to-segment distance.
- Local metric coordinates: equirectangular approximation around each trip's mean longitude/latitude.

## Features
1. GPS-to-road-segment perpendicular distance (m).
2. Absolute difference between GPS movement heading and road bearing (radians).
3. GPS speed (m/s).
4. Time delta (s).
5. Road-segment length (m).
6. Road bearing sine and cosine.
7. GPS heading sine and cosine.
8. Minimum distance to either road endpoint (m).

## Label caveat
The `.route` file is a sequence of road arc IDs, not a per-observation label file. For this baseline, each sampled GPS observation is assigned the spatially nearest segment among the unique route arcs; candidates matching that arc are positive. This is an approximate weak-label construction, not verified temporal alignment. Candidate miss rate is measured and reported. Do not describe these labels as exact GPS-to-road ground truth.

## Validation protocol
Five distinct classifiers were trained. A deterministic 80/20 **trip-group holdout** was used within the 80 training trips (16 validation trips), with no trip split across the two subsets. The 10 student-test trips were used only for development-stage evaluation. The evaluator-test partition was not read.
