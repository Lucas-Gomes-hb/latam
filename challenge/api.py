from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import os
from challenge.model import DelayModel

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class PredictRequest(BaseModel):
    flights: List[Flight]

app = FastAPI()

# Inicializa e treina o modelo com dados históricos
model = DelayModel()
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(current_dir, "data", "data.csv")
historical_data = pd.read_csv(data_path)
features, target = model.preprocess(data=historical_data, target_column="delay")
model.fit(features=features, target=target)

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    try:
        # Converte os dados da requisição para DataFrame
        flight_dicts = [flight.model_dump() for flight in request.flights]
        data = pd.DataFrame(flight_dicts)
        
        # Validações
        if any(data['MES'] > 12) or any(data['MES'] < 1):
            raise ValueError("Month must be between 1 and 12")
        
        if any(~data['TIPOVUELO'].isin(['I', 'N'])):
            raise ValueError("Invalid flight type")
            
        if any(~data['OPERA'].isin(['Aerolineas Argentinas', 'Grupo LATAM', 'Sky Airline', 'Copa Air', 'Latin American Wings'])):
            raise ValueError("Invalid airline operator")
        
        # Preprocessa os novos dados (sem target_column pois estamos prevendo)
        features = model.preprocess(data)
        
        # Faz a predição
        predictions = model.predict(features)
        
        return {"predict": predictions}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))