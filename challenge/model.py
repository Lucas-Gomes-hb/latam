import pandas as pd
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
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # Cria colunas adicionais necessárias para o modelo.
        data['PeriodDay'] = data['Fecha-I'].apply(self._get_period_day)
        data['HighSeason'] = data['Fecha-I'].apply(self._is_high_season)
        data['MinDiff'] = data.apply(self._get_min_diff, axis=1)

        # Remove colunas irrelevantes e separa features/target, se necessário.
        features = data.drop(columns=['Fecha-I', 'Fecha-O', target_column] if target_column else ['Fecha-I', 'Fecha-O'])
        
        if target_column:
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

    # Métodos auxiliares privados.
    def _get_period_day(self, date: str) -> str:
        """
        Calculate the period of the day from a datetime string.
        """
        date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
        morning_min = datetime.strptime("05:00", '%H:%M').time()
        morning_max = datetime.strptime("11:59", '%H:%M').time()
        afternoon_min = datetime.strptime("12:00", '%H:%M').time()
        afternoon_max = datetime.strptime("18:59", '%H:%M').time()
        evening_min = datetime.strptime("19:00", '%H:%M').time()
        evening_max = datetime.strptime("23:59", '%H:%M').time()
        night_min = datetime.strptime("00:00", '%H:%M').time()
        night_max = datetime.strptime("04:59", '%H:%M').time()
        
        if morning_min <= date_time <= morning_max:
            return 'mañana'
        elif afternoon_min <= date_time <= afternoon_max:
            return 'tarde'
        elif evening_min <= date_time <= evening_max or night_min <= date_time <= night_max:
            return 'noche'

    def _is_high_season(self, fecha: str) -> int:
        """
        Determine if a given date is in the high season.
        """
        fecha_año = int(fecha.split('-')[0])
        fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
        
        range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
        range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
        range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
        range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
        range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
        range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
        range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
        range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)
        
        if (range1_min <= fecha <= range1_max or
            range2_min <= fecha <= range2_max or
            range3_min <= fecha <= range3_max or
            range4_min <= fecha <= range4_max):
            return 1
        return 0

    def _get_min_diff(self, row: pd.Series) -> float:
        """
        Calculate the time difference in minutes between two timestamps.
        """
        fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = (fecha_o - fecha_i).total_seconds() / 60
        return min_diff
