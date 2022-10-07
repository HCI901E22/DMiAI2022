import random
from fastapi import APIRouter
from Model import Model
from Trainer import Trainer
from models.dtos import SentimentAnalysisRequestDto, SentimentAnalysisResponseDto, TrainRequest, TrainResponse

router = APIRouter()


@router.post('/predict', response_model=SentimentAnalysisResponseDto)
def predict_endpoint(request: SentimentAnalysisRequestDto):
    m = Model()
    m.load_model('results')
    ratings = m.forward(request.reviews)
    ratings = [round(r) for r in ratings]
    return SentimentAnalysisResponseDto(scores=ratings)


@router.post('/train', response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    print(request)
    t = Trainer()
    result = t.train(request.data_path, request.save_path)
    return TrainResponse(train_loss=result['train_loss'], train_accuracy=result['train_accuracy'], test_loss=result['test_loss'], test_accuracy=result['test_accuracy'])
