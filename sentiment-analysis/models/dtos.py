from typing import List
from pydantic import BaseModel


class SentimentAnalysisRequestDto(BaseModel):
    reviews: List[str]


class SentimentAnalysisResponseDto(BaseModel):
    scores: List[int]


class TrainRequestDto(BaseModel):
    path: str
    lr: float
    epochs: int
