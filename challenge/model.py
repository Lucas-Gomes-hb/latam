from pydantic import BaseModel
from typing import Tuple, List, Union
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import plot_importance

class DelayModel:
    def __init__(self):
        self._model = None
        self._features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10", 
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        self._scale = 1
        self._target = []
        # Adicionar flag para controlar se o modelo já foi treinado
        self._is_trained = False

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Preprocessa os dados e retorna features e target.
        """
        # Criar cópias para não modificar os dados originais
        data = data.copy()
        
        # Converter datas se necessário
        if 'Fecha-I' in data.columns:
            data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        if 'Fecha-O' in data.columns:
            data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])
            
        # Se o modelo ainda não foi treinado e temos as colunas necessárias, 
        # vamos treinar com estes dados
        if not self._is_trained and 'Fecha-I' in data.columns and 'Fecha-O' in data.columns:
            min_diff = ((data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60)
            target = pd.DataFrame({'delay': (min_diff > 15).astype(int)})
            self._target = target
            self.scale(target, 'delay')
        
        # Criar dummies para features categóricas
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')
        ], axis=1)
        
        # Selecionar apenas as top features
        features = features[self._features]
        
        # Se o modelo ainda não foi treinado e temos dados suficientes,
        # vamos treinar automaticamente
        if not self._is_trained and self._target is not None and len(self._target) > 0:
            self.fit(features, self._target)
            self._is_trained = True
        
        if target_column:
            return features, self._target
        
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Treina o modelo usando as features selecionadas e balanceamento de classes.
        """
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=self._scale)
        self._model.fit(features, target.values.ravel())
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Faz previsões com os dados fornecidos.
        """
        
        predictions = self._model.predict(features)
        return predictions.tolist()
    
    def scale(self, y_train, target_column):
        n_y0 = len(y_train[y_train[target_column] == 0])
        n_y1 = len(y_train[y_train[target_column] == 1])
        self._scale = n_y0/n_y1
    
class FlightData(BaseModel):
    Fecha_I: str
    Fecha_O: str
    DIANOM: str
    TIPOVUELO: str
    OPERA: str
    SIGLAORI: str
    SIGLADES: str


class PredictRequest(BaseModel):
    flights: List[FlightData]


class PredictResponse(BaseModel):
    predictions: List[int]
