import random
from fastapi import APIRouter
import torch
from Model import Model
from Trainer import Trainer
from models.dtos import SentimentAnalysisRequestDto, SentimentAnalysisResponseDto, TrainRequestDto

router = APIRouter()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
model.load()


@router.post('/predict', response_model=SentimentAnalysisResponseDto)
def predict_endpoint(request: SentimentAnalysisRequestDto):
    # predicted_scores = [random.randint(1,5) for _ in request.reviews]

    response = SentimentAnalysisResponseDto(
        scores=model.forward(request.reviews)
    )

    return response


@router.get('/load/{path}')
def load_endpoint(path: str):
    model.load(path)
    return "Model loaded"


@router.get('/save/{path}')
def save_endpoint(path: str):
    torch.save(model.state_dict(), path)


@router.post('/train')
def train_endpoint(request: TrainRequestDto):
    t = Trainer(model, request.path, 64)
    t.train(request.lr, request.epochs)
