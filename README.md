# PathSnap – GPS Map Matching Using Machine Learning

PathSnap is a machine-learning-based GPS map-matching system that identifies the most likely road segments corresponding to noisy GPS observations. It combines a tuned machine learning classifier with a Hidden Markov Model (HMM) and the Viterbi algorithm to reconstruct more consistent road routes.

The project also provides a FastAPI backend and a frontend dashboard for submitting trips and visualizing predicted routes.

## Features

- **GPS preprocessing:** Cleans and processes GPS trajectory data.
- **Candidate generation:** Identifies possible road segments for each GPS observation.
- **Feature engineering:** Extracts distance, heading, speed, time difference, and road-network features.
- **Machine learning:** Evaluates multiple classification models to identify a suitable candidate-selection model.
- **Bayesian hyperparameter optimization:** Uses Optuna with the Tree-structured Parzen Estimator (TPE) sampler to tune the HistGradientBoosting classifier.
- **HMM-based map matching:** Uses emission and transition probabilities to model road-segment sequences.
- **Viterbi decoding:** Finds the most likely sequence of road segments.
- **FastAPI backend:** Exposes endpoints for trip retrieval and route prediction.
- **Interactive frontend:** Allows users to select trips, run predictions, and visualize results.

## System Architecture

The system follows a multi-stage pipeline:

1. GPS trajectory preprocessing
2. Candidate road-segment generation
3. Feature extraction
4. Machine learning model training
5. Bayesian hyperparameter optimization
6. HMM-based sequence decoding
7. FastAPI prediction service
8. Frontend route visualization

## Machine Learning Approach

Multiple classification algorithms were evaluated, including:

- Logistic Regression
- Decision Tree
- Random Forest
- Extra Trees
- HistGradientBoosting

HistGradientBoosting was selected as the candidate classification model based on validation performance.

### Bayesian Hyperparameter Optimization

Optuna was used with the TPE sampler to search for improved model hyperparameters.

The optimization explored parameters such as:

- `max_iter`
- `max_leaf_nodes`
- `learning_rate`
- `min_samples_leaf`
- `l2_regularization`

The tuned model achieved the following validation results in the tuning experiment:

| Metric | Baseline | Tuned |
|---|---:|---:|
| Accuracy | 89.73% | 90.12% |
| Precision | 88.71% | 88.90% |
| Recall | 80.20% | 81.27% |
| F1 Score | 84.24% | 84.91% |

The tuned model improved the validation F1 score by **0.67 percentage points**.

## HMM and Viterbi Algorithm

The machine learning classifier produces candidate road-segment probabilities for each GPS observation. These probabilities are used as emission information in the Hidden Markov Model.

The HMM considers:

- **Observations:** GPS coordinates.
- **Hidden states:** Candidate road segments.
- **Emission probabilities:** Classifier-generated candidate probabilities.
- **Transition probabilities:** Road-network connectivity and distance consistency.

The Viterbi algorithm uses dynamic programming to identify the most likely sequence of road segments across the trajectory.

This helps produce a more continuous and plausible route than selecting the highest-probability candidate independently for every observation.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Hyperparameter Optimization | Optuna |
| Sequence Modelling | Hidden Markov Model |
| Decoding Algorithm | Viterbi Algorithm |
| Backend | FastAPI |
| API Server | Uvicorn |
| Frontend | Streamlit |
| Map Visualization | Folium |
| Data Processing | Pandas, NumPy |
| Version Control | Git and GitHub |

## Project Structure

```text
ML-Pathsnap-Hackathon/
│
├── backend/
│   ├── main.py
│   ├── schemas.py
│   └── inference.py
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── components.py
│   └── map_visualization.py
│
├── src/
│   ├── pipeline.py
│   ├── tuning.py
│   └── hmm_decoder.py
│
├── models/
│   └── tuned_model.joblib
│
├── results/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Mxyukh10/ML-Pathsnap-Hackathon.git
cd ML-Pathsnap-Hackathon
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```powershell
venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the Backend

From the project root:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

The backend will be available at:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

### 5. Start the Frontend

Configure the frontend API URL in the frontend `.env` file:

```env
API_BASE_URL=http://localhost:8000
```

Then start Streamlit:

```bash
streamlit run frontend/app.py
```

Open the local URL displayed in the terminal.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Root endpoint |
| GET | `/health` | Checks backend health |
| GET | `/trips` | Retrieves available trip IDs |
| POST | `/predict` | Predicts the route for a selected trip |

### Example Prediction Request

**POST** `/predict`

```json
{
  "trip_id": "00000056"
}
```

The response includes:

- Trip ID
- Prediction status
- Route coordinates
- Predicted road-segment IDs
- Evaluation metrics
- Inference time

## Important Notes

- The trained model must be available in the `models/` directory before running inference.
- The required dataset and trip files must be placed in the expected data directory structure.
- The frontend communicates with the backend through the configured API URL.
- When running the backend and frontend on separate computers, both devices should be connected to the same network, and the frontend API URL should use the backend computer's LAN IP address.
- Ground-truth evaluation data should remain separate from prediction inputs.

## Limitations

- Training labels are approximate and are derived from the supplied route information.
- Hyperparameter optimization uses a fixed group-based training and validation split rather than full cross-validation.
- Route quality depends on GPS observation quality, candidate generation, and road-network connectivity.
- The current visualization uses the returned GPS coordinates and does not represent independently reconstructed road-centerline geometry.

## Future Improvements

- Improve candidate generation and feature engineering.
- Evaluate the model using additional group-based validation splits.
- Improve transition probability modelling.
- Incorporate more detailed road-network geometry.
- Optimize inference speed for larger trajectories.
- Extend the system to support real-time GPS streams.

## Authors

Developed as part of a machine learning hackathon project.

**Project:** PathSnap – GPS Map Matching Using Machine Learning

## Project Workflow

The PathSnap system follows a sequence of processing steps to convert noisy GPS observations into more consistent road routes.

1. **GPS Preprocessing:** Cleans and processes the raw GPS trajectory data.
2. **Candidate Generation:** Identifies possible road segments corresponding to each GPS observation.
3. **Feature Engineering:** Extracts relevant features such as distance, heading, speed and time difference.
4. **Machine Learning Classification:** Evaluates candidate road segments using the trained classification model.
5. **HMM-Based Route Reconstruction:** Uses a Hidden Markov Model to identify a more consistent sequence of road segments.
6. **Route Visualization:** Displays the predicted routes through the application dashboard.

