import os
import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import dill

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

# --- Custom Transformers ---
# These classes define the steps of your pipeline.

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

class NotebookImputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer that EXACTLY replicates the sequential patching strategy
    from the experimental notebook for both fitting and transforming.
    """
    def __init__(self):
        self.mice_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'DaySinceLastOrder', 'OrderCount', 'CouponUsed']
        self.knn_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'DaySinceLastOrder', 'OrderCount', 'CouponUsed', 'NumberOfDeviceRegistered', 'SatisfactionScore', 'CashbackAmount']
        self.median_cols = ['OrderCount', 'CouponUsed']
        self.mice_cols_to_keep = ['Tenure']
        self.knn_cols_to_keep = ['WarehouseToHome', 'HourSpendOnApp', 'DaySinceLastOrder', 'OrderAmountHikeFromlastYear']

    def fit(self, X, y=None):
        logging.info("Fitting the sequential NotebookImputer")
        df_fit = X.copy()

        self.mice_imputer_ = IterativeImputer(max_iter=50, random_state=0)
        mice_imputed_data = self.mice_imputer_.fit_transform(df_fit[self.mice_cols])
        df_mice_imputed = pd.DataFrame(mice_imputed_data, columns=self.mice_cols, index=df_fit.index)
        for col in self.mice_cols_to_keep:
            df_fit[col] = df_mice_imputed[col]

        self.knn_imputer_ = KNNImputer(n_neighbors=5)
        knn_imputed_data = self.knn_imputer_.fit_transform(df_fit[self.knn_cols])
        df_knn_imputed = pd.DataFrame(knn_imputed_data, columns=self.knn_cols, index=df_fit.index)
        for col in self.knn_cols_to_keep:
            df_fit[col] = df_knn_imputed[col]

        self.median_imputer_ = SimpleImputer(strategy='median')
        self.median_imputer_.fit(df_fit[self.median_cols])
        
        return self

    def transform(self, X, y=None):
        logging.info("Applying the sequential NotebookImputer transform")
        df = X.copy()

        mice_imputed_data = self.mice_imputer_.transform(df[self.mice_cols])
        df_mice_imputed = pd.DataFrame(mice_imputed_data, columns=self.mice_cols, index=df.index)
        for col in self.mice_cols_to_keep:
            df[col] = df_mice_imputed[col]

        knn_imputed_data = self.knn_imputer_.transform(df[self.knn_cols])
        df_knn_imputed = pd.DataFrame(knn_imputed_data, columns=self.knn_cols, index=df.index)
        for col in self.knn_cols_to_keep:
            df[col] = df_knn_imputed[col]

        median_imputed_data = self.median_imputer_.transform(df[self.median_cols])
        df[self.median_cols] = median_imputed_data

        return df

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode):
        self.features_to_encode = features_to_encode
        self.target_maps = {}
        self.global_means = {}

    def fit(self, X, y=None):
        y = pd.Series(y, name=TARGET_COLUMN)
        for col in self.features_to_encode:
            if col in X.columns:
                combined = pd.concat([X[col], y], axis=1)
                self.target_maps[col] = combined.groupby(col)[TARGET_COLUMN].mean()
                self.global_means[col] = y.mean()
        return self

    def transform(self, X, y=None):
        X_encoded = X.copy()
        for col in self.features_to_encode:
            if col in X_encoded.columns:
                mapped = X_encoded[col].map(self.target_maps.get(col))
                X_encoded[col] = mapped.fillna(self.global_means.get(col, 0))
        return X_encoded

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_copy = X.copy()
        if "HourSpendOnApp" in X_copy.columns and "NumberOfDeviceRegistered" in X_copy.columns:
            X_copy["Digital_Engagement"] = (X_copy["HourSpendOnApp"].fillna(0) * X_copy["NumberOfDeviceRegistered"].fillna(0))
            X_copy = X_copy.drop(columns=['HourSpendOnApp', 'NumberOfDeviceRegistered'])
        return X_copy

def get_data_transformer_object(schema_config) -> Pipeline:
    try:
        categorical_cols_to_encode = schema_config.get('features_to_encode', [])
        
        # This pipeline encapsulates all reusable transformation steps
        final_pipeline = Pipeline(steps=[
            ('drop_id_cols', DropColumnsTransformer(schema_config.get('drop_columns', []))),
            ('notebook_imputation', NotebookImputer()),
            ('target_encoder', TargetEncoder(features_to_encode=categorical_cols_to_encode)),
            ('feature_engineering', FeatureEngineering()),
            ('scaler', StandardScaler())
        ])

        logging.info("Final Preprocessing Pipeline Initialized")
        return final_pipeline
    except Exception as e:
        raise MyException(e, sys) from e

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config
        self.data_validation_artifact = data_validation_artifact
        self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            # Procedural Step 1: Outlier Removal
            outlier_cols = self._schema_config.get('outlier_removal_col', [])
            logging.info(f"Performing outlier removal on: {outlier_cols}")
            for col in outlier_cols:
                 Q1 = input_feature_train_df[col].quantile(0.25)
                 Q3 = input_feature_train_df[col].quantile(0.75)
                 IQR = Q3 - Q1
                 lower_bound = Q1 - (IQR * 1.5)
                 upper_bound = Q3 + (IQR * 1.5)
                 
                 train_filter = (input_feature_train_df[col] >= lower_bound) & (input_feature_train_df[col] <= upper_bound)
                 input_feature_train_df = input_feature_train_df.loc[train_filter]
                 target_feature_train_df = target_feature_train_df.loc[train_filter]
                 
                 test_filter = (input_feature_test_df[col] >= lower_bound) & (input_feature_test_df[col] <= upper_bound)
                 input_feature_test_df = input_feature_test_df.loc[test_filter]
                 target_feature_test_df = target_feature_test_df.loc[test_filter]
            logging.info("Outlier removal complete.")

            input_feature_train_df.reset_index(drop=True, inplace=True)
            target_feature_train_df.reset_index(drop=True, inplace=True)
            input_feature_test_df.reset_index(drop=True, inplace=True)
            target_feature_test_df.reset_index(drop=True, inplace=True)

            # Step 2: Use the encapsulated pipeline for all other transformations
            preprocessor = get_data_transformer_object(self._schema_config)

            # Fit the entire pipeline on the (outlier-removed) training data
            preprocessor.fit(input_feature_train_df, target_feature_train_df)
            
            # Transform both train and test data
            input_feature_train_arr = preprocessor.transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            # Step 3: Resampling on training data
            smt = SMOTEENN(sampling_strategy="minority", random_state=42)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save the complete pipeline object
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)

            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )
        except Exception as e:
            raise MyException(e, sys) from e