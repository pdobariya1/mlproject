import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_pth = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_tranformer_object(self):
        '''
        This function is responsible for data transforamtion
        '''
        try:
            numerical_columns = ['reading score', 'writing score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]

            num_pipeline = Pipeline(steps=[
                ('Impute', SimpleImputer(strategy='median')),
                ('Scaler', StandardScaler())
            ])
            cat_pipeline = Pipeline(steps=[
                ('Impute', SimpleImputer(strategy='most_frequent')),
                ('OH_Encode', OneHotEncoder()),
                ('Scaler', StandardScaler(with_mean=False))
            ])

            logging.info(f"Numerical Columns : {numerical_columns}")
            logging.info(f"Categorical Columns : {categorical_columns}")

            preprocessor = ColumnTransformer([
                ("num pipeline", num_pipeline, numerical_columns),
                ("cat pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformaton(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading the train and test file')

            preprocessing_obj = self.get_data_tranformer_object()

            target_column_name = 'math score'
            numerical_columns = ['reading score', 'writing score']

            # Divide the train dataset to independent & dependent feature 
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Divide the test dataset to independent & dependent feature
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing on train and test dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_pth,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_pth
            )
        
        except Exception as e:
            raise CustomException(e, sys)
