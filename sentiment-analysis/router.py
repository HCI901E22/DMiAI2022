import random
from fastapi import APIRouter
import torch
from Model import Model
from models.dtos import SentimentAnalysisRequestDto, SentimentAnalysisResponseDto

router = APIRouter()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)


@router.post('/predict', response_model=SentimentAnalysisResponseDto)
def predict_endpoint(request: SentimentAnalysisRequestDto):
    # predicted_scores = [random.randint(1,5) for _ in request.reviews]

    response = SentimentAnalysisResponseDto(
        scores=model.forward(request.reviews)
    )

    return response


@router.get('/load')
def load_endpoint(path: str):
    model.load(path)
    return "Model loaded"

@router.get('/save')
def save_endpoint(path: str):
    torch.save(model.state_dict(), path)

@router.get('/train')
def train_endpoint(path: str):

