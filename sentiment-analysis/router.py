import random
from fastapi import APIRouter
import torch
from Model import Model, preprocess_data
from Trainer import Trainer
from models.dtos import SentimentAnalysisRequestDto, SentimentAnalysisResponseDto, TrainRequestDto, PathRequest

router = APIRouter()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
model.load()


@router.post('/predict', response_model=SentimentAnalysisResponseDto)
def predict_endpoint(request: SentimentAnalysisRequestDto):
    # predicted_scores = [random.randint(1,5) for _ in request.reviews]

    vader = [preprocess_data(t) for t in request.reviews]
    response = SentimentAnalysisResponseDto(
        scores=list(model.forward(torch.tensor(vader)))
    )

    return response


@router.post('/load')
def load_endpoint(request: PathRequest):
    model.load_state_dict(torch.load(request.path, map_location=device))
    return "Model loaded"


@router.post('/save')
def save_endpoint(request: PathRequest):
    torch.save(model.state_dict(), request.path)


@router.post('/train')
def train_endpoint(request: TrainRequestDto):
    t = Trainer(model, request.path, 64)
    t.train(request.lr, request.epochs)
