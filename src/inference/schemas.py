import math
from datetime import datetime, timedelta
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


class Record(BaseModel):
    timestamp: datetime
    pm25: float
    DEWP: float
    TEMP: float
    PRES: float
    Iws: float
    Is: float
    Ir: float
    cbwd: Literal["NE", "NW", "SE", "cv"]

    @field_validator("pm25", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir")
    @classmethod
    def validate_finite(cls, v: float) -> float:
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Valor numerico no puede ser NaN ni infinito")
        return v


class PredictRequest(BaseModel):
    records: list[Record]

    @model_validator(mode="after")
    def validate_sequence(self) -> "PredictRequest":
        if not self.records:
            raise ValueError("La lista de registros no puede estar vacia")
        for i in range(1, len(self.records)):
            prev_t = self.records[i - 1].timestamp
            curr_t = self.records[i].timestamp
            if curr_t <= prev_t:
                raise ValueError(
                    "Timestamps deben estar en orden estrictamente creciente"
                )
            if curr_t - prev_t != timedelta(hours=1):
                raise ValueError(
                    "Timestamps deben estar separados exactamente por una hora"
                )
        return self


class PredictResponse(BaseModel):
    prediction_pm25: float
    unit: str = "ug/m3"
    horizon_hours: int
    predicted_for: datetime
    model: dict
