from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import mlflow
import os
import yaml
from contextlib import asynccontextmanager
import traceback

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Charger les paramètres
params_path = os.path.join(BASE_DIR, "params.yaml")
params = yaml.safe_load(open(params_path))
mlflow_params = params["mlflow"]
aws_params = params["aws"]

# Configurer AWS
os.environ["AWS_ACCESS_KEY_ID"] = aws_params["aws_access_key_id"]
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_params["aws_secret_access_key"]
os.environ["AWS_DEFAULT_REGION"] = aws_params["region_name"]

# Configurer MLflow
mlflow.set_tracking_uri(mlflow_params["MLFLOW_TRACKING_URI"])

MODEL_NAME = mlflow_params["model_name"]
MODEL_VERSION = "latest"

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}", flush=True)
    print(f"Chargement du modèle: {MODEL_NAME}/{MODEL_VERSION}", flush=True)
    try:
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{MODEL_NAME}/{MODEL_VERSION}"
        )
        print("Modèle chargé avec succès", flush=True)
    except Exception as e:
        print(f"Erreur chargement modèle: {str(e)}", flush=True)
        traceback.print_exc()
    yield
    print("Arrêt de l'API", flush=True)

app = FastAPI(
    title="Heart Disease Prediction API",
    lifespan=lifespan
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}




 
@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model not loaded"}
    
    # Convertir toutes les valeurs en float
    data = {key: float(value) for key, value in data.items()}
    
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    result = "Maladie cardiaque détectée" if int(prediction[0]) == 1 else "Pas de maladie cardiaque"
    
    return {
        "prediction": int(prediction[0]),
        "message": result
    }   