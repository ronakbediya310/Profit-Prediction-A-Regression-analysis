import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        # init method
        pass
    def predict(self,features):
        try:
            print(type(features))
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')
            logging.info(features)
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            data_scaled=preprocessor.transform(features)
            logging.info(data_scaled)
            pred=model.predict(data_scaled)
            return pred
        
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
        self,
        rd_spend:float,
        administration: float,
        marketing_spend: float,
        state: str, 
    ):
        self.rd_spend = rd_spend
        self.administration = administration   
        self.marketing_spend = marketing_spend
        self.state = state 

    def get_data_as_dataframe(self):
        try:
            input_dict = {
                'R&D Spend': [self.rd_spend],
                'Administration': [self.administration],
                'Marketing Spend': [self.marketing_spend],
                'State': [self.state]
            }
            df = pd.DataFrame(input_dict)
            print(df.head())
            return df

        except Exception as e:
            raise CustomException(e, sys)
