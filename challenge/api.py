import fastapi
import pandas as pd

from challenge.model import DelayModel, PredictRequest

app = fastapi.FastAPI()
model = DelayModel()

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    try:
        # Converter os dados recebidos para DataFrame
        flight_dicts = [flight.model_dump() for flight in request.flights]
        data = pd.DataFrame(flight_dicts)

        # Pré-processar os dados
        features = model.preprocess(data)

        # Fazer as predições
        predictions = model.predict(features)

        # Retornar as predições como resposta
        return {"predictions": predictions}
    except Exception as e:
        # Retornar erro em caso de exceções
        raise fastapi.HTTPException(status_code=500, detail=str(e))