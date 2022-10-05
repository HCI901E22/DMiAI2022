import random
from fastapi import APIRouter

from Model import Model
from models.dtos import SentimentAnalysisRequestDto, SentimentAnalysisResponseDto

router = APIRouter()

model = Model()


@router.post('/predict', response_model=SentimentAnalysisResponseDto)
def predict_endpoint(request: SentimentAnalysisRequestDto):
    # predicted_scores = [random.randint(1,5) for _ in request.reviews]

    response = SentimentAnalysisResponseDto(
        scores=model.forward(request.reviews)
    )

    return response


@router.get('/load')
def load_endpoint():
    model.load("")
    return "Model loaded"
