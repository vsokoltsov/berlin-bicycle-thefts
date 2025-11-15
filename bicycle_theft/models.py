import re
from datetime import date
from datetime import date as DateType
from pydantic import BaseModel, field_validator, field_serializer

class PredictItem(BaseModel):
    lor: str
    date: date

    @field_validator("lor", mode="before")
    def _norm_lor(cls, v: str) -> str:
        s = re.sub(r"\D", "", str(v))
        return s.zfill(8)

class PredictionResult(BaseModel):
    lor: str
    date: date
    rate_hat_per_1000: float
    count_hat: float
    count_int: int

    @field_serializer("date", when_used="json")
    def _ser_date(self, v: DateType) -> str:
        return v.strftime("%Y-%m-%d")