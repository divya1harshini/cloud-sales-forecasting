from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
from prophet import Prophet
import io

app = FastAPI(title="Cloud Sales Prediction API")

@app.get("/")
def home():
    return {"message": "Cloud Sales Prediction API Running Successfully"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # ---------- FILE FORMAT CHECK ----------
    filename = file.filename.lower()

    try:
        if filename.endswith(".csv"):
            contents = await file.read()
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        elif filename.endswith(".xlsx"):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))

        elif filename.endswith(".json"):
            contents = await file.read()
            df = pd.read_json(io.BytesIO(contents))

        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File reading error: {str(e)}")

    # ---------- COLUMN VALIDATION ----------
    if len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="File must contain at least 2 columns (Date and Sales)")

    # Rename first two columns to Prophet format
    df = df.iloc[:, :2]
    df.columns = ["ds", "y"]

    try:
        df["ds"] = pd.to_datetime(df["ds"])
        df["y"] = pd.to_numeric(df["y"])
    except:
        raise HTTPException(status_code=400, detail="Invalid Date or Sales column format")

    # ---------- MODEL TRAIN ----------
    try:
        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)

        result = forecast[["ds", "yhat"]].tail(30)

        return {
            "status": "success",
            "forecast_next_30_days": result.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")