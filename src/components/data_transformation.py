from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
import sys
import os
import numpy as np 
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
## Data Transformation Config:
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')





## Data Transformation
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            
            logging.info("Data Transformation Component starts")
            categorical_columns=['State']
            numerical_columns=['R&D Spend', 'Administration', 'Marketing Spend']
            
            ## Numerical pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('scaler',StandardScaler())
                ]
            )

            ## categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
                    ('label',OrdinalEncoder()),
                     ('scaler',StandardScaler())
                ]
            )
            
            processor=ColumnTransformer([
                        ('num_pipeline',num_pipeline,numerical_columns),
                        ('cat_pipeline',cat_pipeline,categorical_columns)
            ]) 
            logging.info('Data Transformation done')
            return processor      
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation_object(self,train_data_path,test_data_path):
       try:
            ## reading dataframes:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)
            
            logging.info("Obtaining Preprocessor obj")
            preprocessing_obj=self.get_data_transformation_object()
            
            logging.info("Spliting into independent and dependent features")
            
            drop_columns=['Profit']
            target_column='Profit'
            
            input_feature_train_df=train_df.drop(drop_columns,axis=1)
            logging.info(input_feature_train_df.head())
            target_feature_train_df=train_df[target_column]
            
            input_feature_test_df=test_df.drop(drop_columns,axis=1)
            target_feature_test_df=test_df[target_column]
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info(train_arr)
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                
            )
            
            logging.info("preprocessor.pickle created")
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
       except Exception as e:
           raise CustomException(e,sys)