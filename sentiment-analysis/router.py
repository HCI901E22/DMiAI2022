import random
from fastapi import APIRouter
import torch
from Model import Model
from Trainer import Trainer
from models.dtos import SentimentAnalysisRequestDto, SentimentAnalysisResponseDto, TrainRequestDto, PathRequest

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


@router.post('/load')
def load_endpoint(request: PathRequest):
    model.load(request.path)
    return "Model loaded"


@router.post('/save')
def save_endpoint(request: PathRequest):
    torch.save(model.state_dict(), request.path)


@router.post('/train')
def train_endpoint(request: TrainRequestDto):
    t = Trainer(model, request.path, 64)
    t.train(request.lr, request.epochs)
