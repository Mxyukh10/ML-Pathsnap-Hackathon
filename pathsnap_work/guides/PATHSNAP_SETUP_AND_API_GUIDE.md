# PathSnap - Setup and API Guide

## 1. Project Overview

PathSnap is a GPS map-matching system that identifies likely road segments for noisy GPS observations.

The project combines:
- GPS trajectory preprocessing
- Road-segment candidate generation
- Feature engineering
- Machine-learning classification
- Hidden Markov Model (HMM)
- Viterbi sequence decoding
- FastAPI backend for trip prediction

## 2. Repository Structure

The relevant project files are organized as follows:

```text
ML-Pathsnap-Hackathon/
|
+-- pathsnap_work/
    |
    +-- data/
    |   +-- Student folder/
    |       +-- train/
    |       +-- student_test/
    |
    +-- guides/
    |
    +-- project/
        +-- backend/
        |   +-- main.py
        |   +-- schemas.py
        |   +-- inference.py
        |
        +-- models/
        |   +-- pathsnap_model.joblib
        |   +-- tuned_model.joblib
        |
        +-- results/
        |
        +-- src/
        |   +-- pipeline.py
        |   +-- tuning.py
        |   +-- hmm_decoder.py
        |
        +-- requirements.txt
        +-- run_windows.bat
        +-- demo_predict.py
        +-- README.md
```

## 3. Dataset Organization

The pipeline expects the extracted dataset under:

`pathsnap_work/data/Student folder/`

The expected partitions are:

```text
Student folder/
+-- train/
|   +-- <trip_id>/
|       +-- trip files
|
+-- student_test/
    +-- <trip_id>/
        +-- trip files
```

The project README describes each trip as containing track, nodes, arcs, and route files.

The backend searches for the student-test partition and lists its trip folders.

## 4. Environment Setup

Open PowerShell in the repository root.

Move into the project directory:

```powershell
cd pathsnap_work\project
```

Create a virtual environment:

```powershell
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the following command for the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

## 5. Install Dependencies

Install the packages listed in the project's requirements file:

```powershell
python -m pip install -r requirements.txt
```

The listed dependencies are:

- NumPy
- Pandas
- SciPy
- Scikit-learn
- Joblib
- FastAPI
- Uvicorn
- Optuna
- Folium
- Matplotlib

## 6. Training Pipeline

From the project directory, the documented quick pipeline command is:

```powershell
python src/pipeline.py --quick
```

The Windows batch script `run_windows.bat` performs dependency installation and runs this quick pipeline.

The pipeline uses the training partition and evaluates on the student-test partition. The project README states that the evaluator partition is not loaded by this script.

## 7. Starting the FastAPI Backend

The backend application is defined in:

`backend/main.py`

From the project directory, start the API using:

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

When the server starts successfully, the local API address is:

`http://localhost:8000`

Interactive API documentation is available at:

`http://localhost:8000/docs`

## 8. API Endpoints

The backend defines the following endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Returns project and server status |
| GET | `/health` | Checks backend health |
| GET | `/trips` | Lists available student-test trip IDs |
| POST | `/predict` | Predicts a route for a selected trip |

### 8.1 Root Endpoint

Request:

```http
GET /
```

The response contains the project name, running status, and documentation path.

### 8.2 Health Endpoint

Request:

```http
GET /health
```

Expected response:

```json
{
  "status": "healthy"
}
```

### 8.3 List Trips

Request:

```http
GET /trips
```

The response contains the available trip IDs:

```json
{
  "trips": ["00000056"]
}
```

The example trip ID is illustrative; actual results depend on the dataset folders.

### 8.4 Predict a Trip

Request:

```http
POST /predict
```

Example JSON request:

```json
{
  "trip_id": "00000056"
}
```

The trip ID must match an available folder in the student-test partition.

The response schema contains:

| Field | Description |
|---|---|
| `trip_id` | Selected trip identifier |
| `status` | Prediction status |
| `route_coordinates` | GPS coordinates represented as latitude and longitude pairs |
| `predicted_arc_ids` | Predicted road-segment identifiers |
| `metrics` | Evaluation or decoding metrics |
| `inference_time_seconds` | Time taken for inference in seconds |

## 9. Prediction Workflow

The backend prediction process follows these steps:

1. Receive a trip ID through the prediction request.
2. Check whether the trip exists in the student-test partition.
3. Load the tuned model if it has not already been loaded.
4. Call the HMM decoder to process the selected trip.
5. Read the generated route CSV file.
6. Extract latitude and longitude coordinates.
7. Extract predicted arc IDs.
8. Return the prediction results and inference time.

The backend inference module expects the tuned model at:

`models/tuned_model.joblib`

The HMM route output is expected at:

`results/hmm_route_<trip_id>.csv`

## 10. Error Handling

The API handles several error conditions:

| Condition | HTTP status |
|---|---:|
| Unknown or unavailable trip ID | 400 |
| Missing tuned model or required file | 500 |
| Other unexpected prediction errors | 500 |

Check the terminal running Uvicorn for backend error details.

## 11. Separate Demo Prediction Script

The project also contains `demo_predict.py`.

This script:
- Loads `models/pathsnap_model.joblib`.
- Selects the first available student-test trip.
- Generates road candidates.
- Calculates candidate match probabilities.
- Displays the three highest-scoring candidates for each observation.

Run it from the project directory:

```powershell
python demo_predict.py
```

This is a separate demonstration workflow from the FastAPI `/predict` endpoint. The two workflows use different model artifact paths.

## 12. Important Methodology Note

The project README explains that the supplied route file contains a chronological sequence of arc IDs but does not provide explicit labels for every GPS observation.

The baseline therefore creates approximate observation labels through nearest spatial road-segment alignment to the supplied route.

These labels are an approximation and should not be described as verified ground truth.

Candidate generation also records candidate miss rate. This limitation should be considered when interpreting model evaluation results.

## 13. Troubleshooting

### Missing Python package

Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

### Backend module import error

Make sure the terminal is in the project directory:

```powershell
cd pathsnap_work\project
```

Then start Uvicorn using the documented command.

### Tuned model not found

Check that the required file exists:

```text
models/tuned_model.joblib
```

### Student-test partition not found

Check that the dataset directory exists:

```text
../data/Student folder/student_test
```

Also check that the trip folders are extracted correctly.

### Route output not generated

Check the backend terminal for errors from the HMM decoder and verify that the expected route CSV is generated in the results directory.

## 14. Summary

PathSnap provides a machine-learning-based GPS map-matching pipeline and a FastAPI prediction service.

The documented workflow covers environment setup, dependency installation, pipeline execution, API startup, trip listing, route prediction, and common troubleshooting steps.
