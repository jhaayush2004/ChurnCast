import json
import sys
import os

import pandas as pd
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants import SCHEMA_FILE_PATH, DATA_VALIDATION_REPORT_FILE_NAME

class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        :param data_ingestion_artifact: Output reference of data ingestion artifact stage
        :param data_validation_config: configuration for data validation
        """
        try:
            logging.info(f"{'>>' * 20} Data Validation log started. {'<<' * 20}")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> DataFrame:
        """
        Method Name : read_data
        Description : This static method reads a CSV file into a pandas DataFrame.
        Output      : Returns the DataFrame.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def validate_dataset_schema(self, dataframe: DataFrame) -> bool:
        """
        Method Name : validate_dataset_schema
        Description : This method validates the entire schema, checking for column
                      presence and data types.
        Output      : Returns a bool value based on validation results.
        On Failure  : Write an exception log and then raise an exception.
        """
        try:
            validation_status = True
            validation_errors = []

            # Check for column existence and data types
            schema_columns = self._schema_config["columns"]
            dataframe_columns = list(dataframe.columns)
            
            for col, expected_dtype in schema_columns.items():
                if col not in dataframe_columns:
                    validation_status = False
                    validation_errors.append(f"Missing column: {col}")
                    logging.error(f"Missing column: {col}")
                else:
                    actual_dtype = str(dataframe[col].dtype)
                    if actual_dtype != expected_dtype:
                        validation_status = False
                        validation_errors.append(f"Data type mismatch for column '{col}': Expected '{expected_dtype}', but found '{actual_dtype}'.")
                        logging.error(f"Data type mismatch for column '{col}': Expected '{expected_dtype}', but found '{actual_dtype}'.")

            return validation_status
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name : initiate_data_validation
        Description : This method initiates the data validation component for the pipeline.
        Output      : Returns a DataValidationArtifact.
        On Failure  : Write an exception log and then raise an exception.
        """
        try:
            validation_error_msg = ""
            logging.info("Starting data validation")
            train_df, test_df = (
                DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path),
                DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            )

            # Validate the schema for both train and test data
            train_status = self.validate_dataset_schema(dataframe=train_df)
            if not train_status:
                validation_error_msg += "Schema validation failed for the training data. "
            else:
                logging.info("Schema validation passed for the training data.")

            test_status = self.validate_dataset_schema(dataframe=test_df)
            if not test_status:
                validation_error_msg += "Schema validation failed for the testing data. "
            else:
                logging.info("Schema validation passed for the testing data.")

            validation_status = train_status and test_status
#---------------
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_msg,
                validation_report_file_path=self.data_validation_config.validation_report_file_path
            )

            report_dir = os.path.dirname(data_validation_artifact.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(data_validation_artifact.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise MyException(e, sys) from e