import random
from fastapi import APIRouter
from Model import Model
from Trainer import Trainer
from models.dtos import SentimentAnalysisRequestDto, SentimentAnalysisResponseDto, TrainRequest, TrainResponse, LoadRequest

router = APIRouter()

model = None


@router.post('/predict', response_model=SentimentAnalysisResponseDto)
def predict_endpoint(request: SentimentAnalysisRequestDto):
    global model
    if model is None:
        model = Model()
        model.load_model('results')
    ratings = model.forward(request.reviews)
    ratings = [round(r) for r in ratings]
    return SentimentAnalysisResponseDto(scores=ratings)


@router.post('/train', response_model=TrainResponse)
def train(request: TrainRequest) -> TrainResponse:
    print(request)
    t = Trainer(request.model_path)
    result = t.train(request.data_path, request.save_path)
    global model
    model = t.model
    model.save_model(request.save_path)
    return TrainResponse(train_loss=result['train_loss'], train_accuracy=result['train_accuracy'],
                         test_loss=result['test_loss'], test_accuracy=result['test_accuracy'])

@router.post('/load')
def load(request: LoadRequest):
    global model
    model = Model()
    model.load_model(request.path)
    return "Model loaded"
