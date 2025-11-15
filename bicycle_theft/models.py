from typing import List, Annotated
import re
import datetime as dt
from datetime import date as DateType
from pydantic import Field, BaseModel, field_validator, field_serializer
from pydantic.types import StringConstraints

LOR_ANNOTATION = Annotated[str, StringConstraints(pattern=r"^\d{8}$")]


class PredictItem(BaseModel):
    lor: LOR_ANNOTATION = Field(description="8-digit LOR code", examples=["10100205"])
    date: dt.date = Field(
        description="Target date (YYYY-MM-DD)", examples=["2025-11-15"]
    )

    @field_validator("lor", mode="before")
    def _norm_lor(cls, v: str) -> str:
        s = re.sub(r"\D", "", str(v))
        return s.zfill(8)


class PredictionResult(BaseModel):
    lor: str
    date: dt.date
    rate_hat_per_1000: float
    count_hat: float
    count_int: int

    @field_serializer("date", when_used="json")
    def _ser_date(self, v: DateType) -> str:
        return v.strftime("%Y-%m-%d")


class PredictionRequest(BaseModel):
    items: List[PredictItem]


class PredictionResponse(BaseModel):
    items: List[PredictionResult]
