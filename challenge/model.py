import pandas as pd

from pydantic  import BaseModel
from datetime import datetime
from typing import Tuple, Union, List
from sklearn.ensemble import RandomForestClassifier

class DelayModel:

    def __init__(self):
        self._model = RandomForestClassifier()  # Modelo inicializado.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or prediction.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target if target_column is provided.
            or
            pd.DataFrame: features only.
        """
        # Create high_season column
        data['high_season'] = data['Fecha-I'].apply(self.is_high_season)

        # Create min_diff column (in minutes)
        data['min_diff'] = data.apply(self.get_min_diff, axis=1)

        # Create period_day column
        data['period_day'] = data['Fecha-I'].apply(self.get_period_day)

        # Create delay column
        data['delay'] = (data['min_diff'] > 15).astype(int)

        # Columns to keep for training/prediction
        columns_to_keep = [
            'high_season', 'min_diff', 'period_day', 'DIANOM', 
            'TIPOVUELO', 'OPERA', 'SIGLAORI', 'SIGLADES'
        ]

        features = data[columns_to_keep]

        if target_column:
            if target_column not in data.columns:
                raise KeyError(f"Target column '{target_column}' not found in data.")
            target = data[target_column]
            return features, target

        return features


    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        self._model.fit(features, target)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        return self._model.predict(features).tolist()

    def get_period_day(self, fecha: str) -> str:
        hour = pd.to_datetime(fecha).hour
        if 5 <= hour <= 11:
            return 'morning'
        elif 12 <= hour <= 18:
            return 'afternoon'
        else:
            return 'night'

    def is_high_season(self, fecha: str) -> int:
        date = pd.to_datetime(fecha)
        if ((date.month == 12 and date.day >= 15) or
            (date.month == 1 or date.month == 2) or
            (date.month == 3 and date.day <= 3) or
            (date.month == 7 and date.day >= 15 and date.day <= 31) or
            (date.month == 9 and date.day >= 11 and date.day <= 30)):
            return 1
        return 0


    def get_min_diff(self, row: pd.Series) -> int:
        date_i = pd.to_datetime(row['Fecha-I'])
        date_o = pd.to_datetime(row['Fecha-O'])
        return int((date_o - date_i).total_seconds() / 60)


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