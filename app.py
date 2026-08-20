from fastapi import FastAPI
from datetime import datetime
import pandas as pd
import joblib
from pydantic import BaseModel, Field
import uvicorn

# load our model and preprocessor
Nova_Model = joblib.load('notebook/Nova_Model.joblib')
preprocessor = joblib.load('notebook/preprocessor.joblib')

# Fail loudly at startup, not on the first request, if these aren't
# real fitted objects — this is exactly the bug that caused the
# 'list' object has no attribute 'transform' error.
assert hasattr(preprocessor, 'transform'), (
    "preprocessor.joblib did not load as a fitted transformer "
    f"(got {type(preprocessor)}). Re-run train_and_save_for_deployment.py."
)
assert hasattr(Nova_Model, 'predict'), (
    f"Nova_Model.joblib did not load as a fitted model (got {type(Nova_Model)})."
)

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    home_country: str = Field(example="us")
    source_currency: str = Field(example="usd")
    dest_currency: str = Field(example="cad")
    channel: str = Field(example="web")
    amount_src: float = Field(example=278.10)
    amount_usd: float = Field(example=278.10)
    fee: float = Field(example=4.20)
    exchange_rate_src_to_dest: float = Field(example=1.25)
    new_device: bool = Field(example=False)
    device_id: str = Field(example="0")
    ip_address: str = Field(example="0.0")
    customer_id: str = Field(example="00042")
    location_mismatch: bool = Field(example=False)
    ip_risk_score: float = Field(example=0.12)
    kyc_tier: str = Field(example="standard")
    account_age_days: float = Field(example=263)
    device_trust_score: float = Field(example=0.52)
    chargeback_history_count: int = Field(example=0)
    risk_score_internal: float = Field(example=0.22)
    txn_velocity_1h: int = Field(example=0)
    txn_velocity_24h: int = Field(example=0)
    corridor_risk: float = Field(example=0.0)
    timestamp: datetime = Field(example="2026-08-01T18:30:00")


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    df = pd.DataFrame([transaction.model_dump()])

    # No manual bucketing/encoding here on purpose — the preprocessor
    # object already knows exactly how to do this because it's the
    # SAME object that was fit during training. Reimplementing bucket
    # edges/labels by hand here (as the previous version did) is what
    # causes train/serve skew even when nothing throws an error.
    df_processed = preprocessor.transform(df)

    pred = int(Nova_Model.predict(df_processed)[0])
    proba = float(Nova_Model.predict_proba(df_processed)[0][1])

    return {
        "is_fraud": pred,
        "fraud_probability": proba,
        "label": "Fraud" if pred == 1 else "Legit",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)