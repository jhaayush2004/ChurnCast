import sys
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

# Importing constants and pipeline modules from your project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import ChurnData, ChurnPredictor
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (e.g., CSS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML pages
templates = Jinja2Templates(directory='templates')

# Configure Cross-Origin Resource Sharing (CORS) to allow all origins
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    """
    DataForm class to handle and process incoming form data for churn prediction.
    This class defines all the customer-related attributes expected from the web form.
    """
    def __init__(self, request: Request):
        self.request: Request = request
        self.Tenure: Optional[float] = None
        self.CityTier: Optional[int] = None
        self.WarehouseToHome: Optional[float] = None
        self.HourSpendOnApp: Optional[float] = None
        self.NumberOfDeviceRegistered: Optional[int] = None
        self.SatisfactionScore: Optional[int] = None
        self.NumberOfAddress: Optional[int] = None
        self.Complain: Optional[int] = None
        self.OrderAmountHikeFromlastYear: Optional[float] = None
        self.CouponUsed: Optional[float] = None
        self.OrderCount: Optional[float] = None
        self.DaySinceLastOrder: Optional[float] = None
        self.CashbackAmount: Optional[float] = None
        self.Gender: Optional[str] = None
        self.PreferedOrderCat: Optional[str] = None
        self.MaritalStatus: Optional[str] = None
        self.PreferredLoginDevice: Optional[str] = None
        self.PreferredPaymentMode: Optional[str] = None

    async def get_churn_data(self):
        """
        Asynchronously retrieves and assigns form data to the class attributes.
        """
        form = await self.request.form()
        self.Tenure = form.get("Tenure")
        self.CityTier = form.get("CityTier")
        self.WarehouseToHome = form.get("WarehouseToHome")
        self.HourSpendOnApp = form.get("HourSpendOnApp")
        self.NumberOfDeviceRegistered = form.get("NumberOfDeviceRegistered")
        self.SatisfactionScore = form.get("SatisfactionScore")
        self.NumberOfAddress = form.get("NumberOfAddress")
        self.Complain = form.get("Complain")
        self.OrderAmountHikeFromlastYear = form.get("OrderAmountHikeFromlastYear")
        self.CouponUsed = form.get("CouponUsed")
        self.OrderCount = form.get("OrderCount")
        self.DaySinceLastOrder = form.get("DaySinceLastOrder")
        self.CashbackAmount = form.get("CashbackAmount")
        self.Gender = form.get("Gender")
        self.PreferedOrderCat = form.get("PreferedOrderCat")
        self.MaritalStatus = form.get("MaritalStatus")
        self.PreferredLoginDevice = form.get("PreferredLoginDevice")
        self.PreferredPaymentMode = form.get("PreferredPaymentMode")


@app.get("/", tags=["prediction"])
async def index(request: Request):
    """
    Renders the main HTML page with the form for user input.
    """
    return templates.TemplateResponse("index.html", {"request": request, "context": "Rendering"})


@app.get("/train")
async def train_route_client():
    """
    Endpoint to initiate the model training pipeline.
    """
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predict_route_client(request: Request):
    """
    Endpoint to receive form data, process it, and make a churn prediction.
    """
    try:
        # Get data from the form
        form = DataForm(request)
        await form.get_churn_data()
        
        # Create a ChurnData object with the correct data types
        churn_data = ChurnData(
            Tenure=float(form.Tenure),
            CityTier=int(form.CityTier),
            WarehouseToHome=float(form.WarehouseToHome),
            HourSpendOnApp=float(form.HourSpendOnApp),
            NumberOfDeviceRegistered=int(form.NumberOfDeviceRegistered),
            SatisfactionScore=int(form.SatisfactionScore),
            NumberOfAddress=int(form.NumberOfAddress),
            Complain=int(form.Complain),
            OrderAmountHikeFromlastYear=float(form.OrderAmountHikeFromlastYear),
            CouponUsed=float(form.CouponUsed),
            OrderCount=float(form.OrderCount),
            DaySinceLastOrder=float(form.DaySinceLastOrder),
            CashbackAmount=float(form.CashbackAmount),
            Gender=form.Gender,
            PreferedOrderCat=form.PreferedOrderCat,
            MaritalStatus=form.MaritalStatus,
            PreferredLoginDevice=form.PreferredLoginDevice,
            PreferredPaymentMode=form.PreferredPaymentMode
        )

        # Convert to DataFrame
        churn_df = churn_data.get_churn_input_data_frame()

        # Initialize the predictor and make a prediction
        model_predictor = ChurnPredictor()
        prediction_result = model_predictor.predict(dataframe=churn_df)

        # Render the same HTML page with the prediction result
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": prediction_result},
        )
    
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": f"Error: {e}"},
        )


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)