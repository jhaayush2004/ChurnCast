import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from typing import Optional
from dataclasses import dataclass
# In your prediction/evaluation script
from src.components.data_transformation import NotebookImputer, TargetEncoder, FeatureEngineering, DropColumnsTransformer
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_object, read_yaml_file
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainerArtifact, DataTransformationArtifact, ModelEvaluationArtifact
from src.entity.s3_estimator import Proj1Estimator
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.components.data_transformation import NotebookImputer, TargetEncoder, FeatureEngineering, DropColumnsTransformer
@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float

class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 20} Model Evaluation log started. {'<<' * 20}")
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.data_transformation_artifact = data_transformation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name : get_best_model
        Description : This function is used to get model from production stage.
        
        Output      : Returns model object if available in s3 storage
        On Failure  : Write an exception log and then raise an exception
        """
        try:
            logging.info("Checking if best model is present in S3 bucket.")
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name, model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                logging.info("Best model found in S3.")
                return proj1_estimator
            logging.info("No best model found in S3.")
            return None
        except Exception as e:
            raise MyException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name : evaluate_model
        Description : This function is used to evaluate the trained model
                      with the production model and choose the best model.
        
        Output      : Returns a boolean value based on validation results
        On Failure  : Write an exception log and then raise an exception
        """
        try:
            logging.info("Loading transformed test data and trained model.")
            
            # Load transformed test data from the data_transformation_artifact
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)
            
            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            trained_model_object = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            
            # The preprocessor object is not needed here as the data is already transformed
            
            trained_model_predictions = trained_model_object.trained_model_object.predict(x_test)
            trained_model_f1_score = f1_score(y_test, trained_model_predictions)
            logging.info(f"F1_Score for trained model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model_object = self.get_best_model()

            if best_model_object is not None:
                logging.info(f"Computing F1_Score for production model...")
                # The best_model_object is a Proj1Estimator, which has its own predict method
                # It should be designed to handle raw data, so we load the raw test data for this step
                raw_test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
                y_hat_best_model = best_model_object.predict(raw_test_df.drop(columns=[TARGET_COLUMN], axis=1))
                best_model_f1_score = f1_score(raw_test_df[TARGET_COLUMN], y_hat_best_model)
                logging.info(f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}")
            else:
                logging.info("No production model found. Trained model is accepted by default.")

            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            is_model_accepted = trained_model_f1_score > tmp_best_model_score
            
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=is_model_accepted,
                difference=trained_model_f1_score - tmp_best_model_score
            )
            logging.info(f"Evaluation result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name : initiate_model_evaluation
        Description : This function is used to initiate all steps of the model evaluation.
        
        Output      : Returns model evaluation artifact.
        On Failure  :  Write an exception log and then raise an exception.
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()

            s3_model_path = self.model_eval_config.s3_model_key_path
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e