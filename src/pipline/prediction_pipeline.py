import sys
import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file
from src.constants import SCHEMA_FILE_PATH
from src.entity.config_entity import ChurnPredictorConfig
from src.entity.s3_estimator import Proj1Estimator

# It's crucial to import your custom transformers here so that the loaded model can be unpickled
from src.components.data_transformation import DropColumnsTransformer, NotebookImputer, TargetEncoder, FeatureEngineering


class ChurnData:
    def __init__(self,
                 Tenure: float,
                 CityTier: int,
                 WarehouseToHome: float,
                 HourSpendOnApp: float,
                 NumberOfDeviceRegistered: int,
                 SatisfactionScore: int,
                 NumberOfAddress: int,
                 Complain: int,
                 OrderAmountHikeFromlastYear: float,
                 CouponUsed: float,
                 OrderCount: float,
                 DaySinceLastOrder: float,
                 CashbackAmount: float,
                 Gender: str,
                 PreferedOrderCat: str,
                 MaritalStatus: str,
                 PreferredLoginDevice: str,
                 PreferredPaymentMode: str
                 ):
        """
        ChurnData constructor.
        Accepts all raw features needed for the churn prediction model.
        """
        try:
            self.Tenure = Tenure
            self.CityTier = CityTier
            self.WarehouseToHome = WarehouseToHome
            self.HourSpendOnApp = HourSpendOnApp
            self.NumberOfDeviceRegistered = NumberOfDeviceRegistered
            self.SatisfactionScore = SatisfactionScore
            self.NumberOfAddress = NumberOfAddress
            self.Complain = Complain
            self.OrderAmountHikeFromlastYear = OrderAmountHikeFromlastYear
            self.CouponUsed = CouponUsed
            self.OrderCount = OrderCount
            self.DaySinceLastOrder = DaySinceLastOrder
            self.CashbackAmount = CashbackAmount
            self.Gender = Gender
            self.PreferedOrderCat = PreferedOrderCat
            self.MaritalStatus = MaritalStatus
            self.PreferredLoginDevice = PreferredLoginDevice
            self.PreferredPaymentMode = PreferredPaymentMode
            # Add placeholders for ID columns that the pipeline expects to drop
            self.CustomerID = "placeholder"
            self._id = "placeholder"

        except Exception as e:
            raise MyException(e, sys) from e

    def get_churn_data_as_dict(self):
        """
        Returns a dictionary representation of the ChurnData.
        This dictionary includes ALL columns the pipeline was trained on.
        """
        try:
            input_data = {
                "CustomerID": [self.CustomerID],
                "Tenure": [self.Tenure],
                "PreferredLoginDevice": [self.PreferredLoginDevice],
                "CityTier": [self.CityTier],
                "WarehouseToHome": [self.WarehouseToHome],
                "PreferredPaymentMode": [self.PreferredPaymentMode],
                "Gender": [self.Gender],
                "HourSpendOnApp": [self.HourSpendOnApp],
                "NumberOfDeviceRegistered": [self.NumberOfDeviceRegistered],
                "PreferedOrderCat": [self.PreferedOrderCat],
                "SatisfactionScore": [self.SatisfactionScore],
                "MaritalStatus": [self.MaritalStatus],
                "NumberOfAddress": [self.NumberOfAddress],
                "Complain": [self.Complain],
                "OrderAmountHikeFromlastYear": [self.OrderAmountHikeFromlastYear],
                "CouponUsed": [self.CouponUsed],
                "OrderCount": [self.OrderCount],
                "DaySinceLastOrder": [self.DaySinceLastOrder],
                "CashbackAmount": [self.CashbackAmount],
                "_id": [self._id]
            }
            return input_data
        except Exception as e:
            raise MyException(e, sys) from e

    def get_churn_input_data_frame(self) -> pd.DataFrame:
        """
        Returns a pandas DataFrame from the ChurnData.
        """
        try:
            churn_input_dict = self.get_churn_data_as_dict()
            return pd.DataFrame(churn_input_dict)
        except Exception as e:
            raise MyException(e, sys) from e


class ChurnPredictor:
    def __init__(self, prediction_pipeline_config: ChurnPredictorConfig = ChurnPredictorConfig()):
        """
        Initializes the ChurnPredictor.
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
            # Load the schema to get the correct column order
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe: pd.DataFrame) -> str:
        """
        Takes a raw DataFrame, enforces the correct column order, and returns the prediction.
        """
        try:
            logging.info("Entered predict method of ChurnPredictor class")
            
            # --- THIS IS THE FIX ---
            # Get the original training column order from the schema file
            all_cols = list(self.schema['columns'].keys())
            # Remove the target variable 'Churn' to get the input feature order
            training_column_order = [col for col in all_cols if col != 'Churn']
            
            # Reorder the incoming dataframe to match the training order
            dataframe = dataframe[training_column_order]

            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)
            
            # Assuming the model returns 1 for Churn and 0 for No Churn
            if result[0] == 1:
                prediction = "Customer will CHURN"
            else:
                prediction = "Customer will NOT CHURN"
            
            return prediction
        
        except Exception as e:
            raise MyException(e, sys)
