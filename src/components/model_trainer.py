from dataclasses import dataclass
import sys
import os
import numpy as np 
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from src.utils import evaluate_model


##Importing all the Machine Learning Models That I am going to use:-
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiate_model_training(self,train_array,test_array):
        logging.info("Train Test Split starts")
        
        try:
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Linear Regression': (LinearRegression()),  
                'Ridge Regression': (Ridge()),
                'Lasso Regression': (Lasso()),
                'Random Forest Regression': (RandomForestRegressor()),
                'Gradient Boosting Regression': (GradientBoostingRegressor()),
                'Support Vector Regression': (SVR()),
                'K-Nearest Neighbors Regression': (KNeighborsRegressor()),
                'Decision Tree Regression': (DecisionTreeRegressor())  
            }
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            
            logging.info(model_report)
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=models[best_model_name]
            logging.info(best_model_name) 
            logging.info(best_model_score) 
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
        except Exception as e:
            raise CustomException(e,sys)