import os
import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import dill

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

# Add this new class to your script
class NotebookImputer(BaseEstimator, TransformerMixin):
    """
    A custom imputer that replicates the sequential patching strategy
    from the experimental notebook.
    """
    def __init__(self):
        # Define the columns for each imputation strategy
        self.mice_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'DaySinceLastOrder', 'OrderCount', 'CouponUsed']
        self.knn_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'DaySinceLastOrder', 'OrderCount', 'CouponUsed', 'NumberOfDeviceRegistered', 'SatisfactionScore', 'CashbackAmount']
        self.median_cols = ['OrderCount', 'CouponUsed']

        # Define which columns to KEEP from each strategy
        self.mice_cols_to_keep = ['Tenure']
        self.knn_cols_to_keep = ['WarehouseToHome', 'HourSpendOnApp', 'DaySinceLastOrder', 'OrderAmountHikeFromlastYear']

    def fit(self, X, y=None):
        logging.info("Fitting the custom NotebookImputer")
        # Create and fit all the imputers on the training data
        self.mice_imputer_ = IterativeImputer(max_iter=50, random_state=0)
        self.mice_imputer_.fit(X[self.mice_cols])

        self.knn_imputer_ = KNNImputer(n_neighbors=5)
        self.knn_imputer_.fit(X[self.knn_cols])

        self.median_imputer_ = SimpleImputer(strategy='median')
        self.median_imputer_.fit(X[self.median_cols])
        
        return self

    def transform(self, X, y=None):
        logging.info("Applying the custom NotebookImputer transform")
        df = X.copy()

        # Step 1: Apply MICE and keep only the specified column
        mice_imputed_data = self.mice_imputer_.transform(df[self.mice_cols])
        df_mice_imputed = pd.DataFrame(mice_imputed_data, columns=self.mice_cols, index=df.index)
        for col in self.mice_cols_to_keep:
            df[col] = df_mice_imputed[col]

        # Step 2: Apply KNN and keep only the specified columns
        knn_imputed_data = self.knn_imputer_.transform(df[self.knn_cols])
        df_knn_imputed = pd.DataFrame(knn_imputed_data, columns=self.knn_cols, index=df.index)
        for col in self.knn_cols_to_keep:
            df[col] = df_knn_imputed[col]

        # Step 3: Apply Median and update its columns
        median_imputed_data = self.median_imputer_.transform(df[self.median_cols])
        df[self.median_cols] = median_imputed_data

        return df
# TargetEncoder and FeatureEngineering classes remain the same
class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, features_to_encode):
        self.features_to_encode = features_to_encode
        self.target_maps = {}
        self.global_means = {}

    def fit(self, X, y=None):
        logging.info("Fitting TargetEncoder...")
        y = pd.Series(y, name=TARGET_COLUMN)
        for col in self.features_to_encode:
            if col in X.columns:
                combined = pd.concat([X[col], y], axis=1)
                mean_target_map = combined.groupby(col)[TARGET_COLUMN].mean()
                self.target_maps[col] = mean_target_map
                self.global_means[col] = y.mean()
        logging.info("TargetEncoder fit complete.")
        return self

    def transform(self, X, y=None):
        logging.info("Transforming data with TargetEncoder...")
        X_encoded = X.copy()
        for col in self.features_to_encode:
            if col in X_encoded.columns:
                mapped = X_encoded[col].map(self.target_maps.get(col))
                # This ensures the output column is numeric
                X_encoded[col] = mapped.fillna(self.global_means.get(col, 0))
        logging.info("TargetEncoder transform complete.")
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


# FINAL CORRECTED PIPELINE BUILDER
# Replace your existing function with this one
def get_data_transformer_object(schema_config) -> Pipeline:
    try:
        logging.info("Building preprocessor pipeline based on notebook strategy.")
        
        categorical_cols_to_encode = schema_config.get('features_to_encode', [])
        
        # This is a clean, sequential pipeline that perfectly mirrors your notebook
        final_pipeline = Pipeline(steps=[
            # Step 1: Apply your custom sequential imputation
            ('notebook_imputation', NotebookImputer()),
            
            # Step 2: Encode categorical features. Receives a DF, works correctly.
            ('target_encoder', TargetEncoder(features_to_encode=categorical_cols_to_encode)),
            
            # Step 3: Engineer features. Receives a DF, works correctly.
            ('feature_engineering', FeatureEngineering()),
            
            # Step 4: Scale all resulting numeric columns.
            ('scaler', StandardScaler())
        ])

        logging.info("Final Preprocessing Pipeline Initialized")
        return final_pipeline

    except Exception as e:
        raise MyException(e, sys) from e


# The DataTransformation class remains the same
class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"{'>>' * 20} Data Transformation log started. {'<<' * 20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def _handle_outliers(self, df: pd.DataFrame, target: pd.Series, column_name: str) -> (pd.DataFrame, pd.Series):
        logging.info(f"Handling outliers for column: {column_name}")
        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + (IQR * 1.5)
        lower_bound = Q1 - (IQR * 1.5)
        filtered_indices = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)].index
        return df.loc[filtered_indices], target.loc[filtered_indices]

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            numerical_cols_for_outliers = self._schema_config.get('outlier_removal_col', [])
            for col in numerical_cols_for_outliers:
                if col in input_feature_train_df.columns:
                     input_feature_train_df, target_feature_train_df = self._handle_outliers(input_feature_train_df, target_feature_train_df, col)
                if col in input_feature_test_df.columns:
                     input_feature_test_df, target_feature_test_df = self._handle_outliers(input_feature_test_df, target_feature_test_df, col)
            logging.info("Outlier removal complete.")

            input_feature_train_df.reset_index(drop=True, inplace=True)
            target_feature_train_df.reset_index(drop=True, inplace=True)
            input_feature_test_df.reset_index(drop=True, inplace=True)
            target_feature_test_df.reset_index(drop=True, inplace=True)
            
            drop_columns = self._schema_config.get('drop_columns', [])
            input_feature_train_df.drop(columns=[c for c in drop_columns if c in input_feature_train_df.columns], inplace=True, errors='ignore')
            input_feature_test_df.drop(columns=[c for c in drop_columns if c in input_feature_test_df.columns], inplace=True, errors='ignore')
            logging.info("Dropped unnecessary columns.")
            
            preprocessor = get_data_transformer_object(self._schema_config)

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df, target_feature_train_df)
            
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority", random_state=42)
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)
            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e