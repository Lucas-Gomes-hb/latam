from typing import Tuple, List, Union
import pandas as pd
import xgboost as xgb

class DelayModel:
    def __init__(self):
        self._model = xgb.XGBClassifier(random_state=1)
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
        self._target = pd.DataFrame([0])
        self._is_trained = False

    def preprocess(self, data: pd.DataFrame, target_column: str = None) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Preprocessa os dados e retorna features e target.
        """
        data = data.copy()
        
        if 'Fecha-I' in data.columns:
            data['Fecha-I'] = pd.to_datetime(data['Fecha-I'])
        if 'Fecha-O' in data.columns:
            data['Fecha-O'] = pd.to_datetime(data['Fecha-O'])
            
        if not self._is_trained and 'Fecha-I' in data.columns and 'Fecha-O' in data.columns:
            min_diff = ((data['Fecha-O'] - data['Fecha-I']).dt.total_seconds() / 60)
            target = pd.DataFrame({'delay': (min_diff > 15).astype(int)})
            self._target = target
            self.scale(target, 'delay')
        
        # Criar features dummy com todas as colunas necessárias
        features = pd.DataFrame()
        
        # OPERA dummies
        opera_dummies = pd.get_dummies(data['OPERA'], prefix='OPERA')
        for col in ['OPERA_Latin American Wings', 'OPERA_Grupo LATAM', 'OPERA_Sky Airline', 'OPERA_Copa Air']:
            if col not in opera_dummies.columns:
                opera_dummies[col] = 0
        
        # TIPOVUELO dummies
        tipovuelo_dummies = pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO')
        if 'TIPOVUELO_I' not in tipovuelo_dummies.columns:
            tipovuelo_dummies['TIPOVUELO_I'] = 0
        
        # MES dummies
        mes_dummies = pd.get_dummies(data['MES'], prefix='MES')
        for mes in [4, 7, 10, 11, 12]:
            col_name = f'MES_{mes}'
            if col_name not in mes_dummies.columns:
                mes_dummies[col_name] = 0
        
        # Concatenar todas as features
        features = pd.concat([
            opera_dummies,
            tipovuelo_dummies,
            mes_dummies
        ], axis=1)
        
        # Selecionar apenas as features necessárias na ordem correta
        features = features[self._features]
        
        if not self._is_trained and self._target is not None and len(self._target) > 0:
            self.fit(features, self._target)
            self._is_trained = True
        
        if target_column:
            return features, self._target
        
        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=self._scale)
        self._model.fit(features, target.values.ravel())
        self._is_trained = True

    def predict(self, features: pd.DataFrame) -> List[int]:
        predictions = self._model.predict(features)
        return predictions.tolist()
    
    def scale(self, y_train, target_column):
        n_y0 = len(y_train[y_train[target_column] == 0])
        n_y1 = len(y_train[y_train[target_column] == 1])
        self._scale = n_y0/n_y1