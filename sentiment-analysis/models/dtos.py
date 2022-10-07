from typing import List
from pydantic import BaseModel


class SentimentAnalysisRequestDto(BaseModel):
    reviews: List[str]


class SentimentAnalysisResponseDto(BaseModel):
    scores: List[float]


class TrainRequestDto(BaseModel):
    path: str
    lr: float
    epochs: int


class PathRequest(BaseModel):
    path: str