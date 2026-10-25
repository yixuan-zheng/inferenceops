from pydantic import BaseModel


class Prediction(BaseModel):
    class_id: int
    label: str
    probability: float


class Timings(BaseModel):
    accumulation_ms: float
    batch_size: int
    preprocessing_ms: float
    preprocessing_wait_ms: float
    inference_wait_ms: float
    inference_ms: float
    server_total_ms: float


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    predictions: list[Prediction]
    timings: Timings
