from typing import List
from pydantic import BaseModel


class SentimentAnalysisRequestDto(BaseModel):
    reviews: List[str]


class SentimentAnalysisResponseDto(BaseModel):
    scores: List[int]


class TrainRequest(BaseModel):
    data_path: str
    save_path: str
    model_path: str = None


class TrainResponse(BaseModel):
    train_loss: float
    train_accuracy: float
    test_loss: float
    test_accuracy: float
