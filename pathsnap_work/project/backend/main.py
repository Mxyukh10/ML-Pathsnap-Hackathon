
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas import PredictionRequest, PredictionResponse
from backend.inference import list_trips, predict_trip

app = FastAPI(
    title="PathSnap GPS Map Matching API",
    description="GPS trajectory map matching using HistGradientBoosting and HMM/Viterbi",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "project": "PathSnap",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/trips")
def trips():
    try:
        return {"trips": list_trips()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        return predict_trip(request.trip_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
