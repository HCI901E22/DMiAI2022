import cv2
import random
import base64
import numpy as np
from typing import List
from loguru import logger
from fastapi import APIRouter, Request
from models.dtos import PredictRequestDto, PredictResponseDto, BoundingBoxClassification, LoadRequest
from PIL import Image
from io import BytesIO

import os



router = APIRouter()
model = None


@router.post('/predict', response_model=PredictResponseDto)
def predict_endpoint(request: PredictRequestDto):
    img: np.ndarray = decode_request(request)

    dummy_bounding_boxes = predict(img)
    response = PredictResponseDto(
        boxes=dummy_bounding_boxes
    )

    return response

@router.post('/load')
def load_model(request: LoadRequest):
    from keras_retinanet_main.keras_retinanet.models import load_model
    global model
    model = load_model(request.path, backbone_name = 'resnet101')
    return "Model loded"


def decode_request(request: PredictRequestDto) -> np.ndarray:
    encoded_img: str = request.img
    im = Image.open(BytesIO(base64.b64decode(encoded_img)))
    #path = "eval-images/reeee" + str(random.randint(0,10000)) + ".png"
    #im.save(path, 'PNG')
    np_img = np.fromstring(base64.b64decode(encoded_img), np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_ANYCOLOR)


def predict(img: np.ndarray) -> List[BoundingBoxClassification]:
    logger.info(f'Recieved image: {img.shape}')
    
    #import os
    #from keras_retinanet_main.keras_retinanet.models import load_model
    #model = load_model('snapshots/mymodelv02.h5', backbone_name = 'resnet101')
    # load retinanet model
    result: List[BoundingBoxClassification] = []
    
    global model
    boxes, scores, labels = model.predict(np.expand_dims(img, axis=0))
    count = 0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        count = count + 1
        if score < 0.5:
            print("breaking after " + str(count))
            break
        b = BoundingBoxClassification(class_id=label, min_x=box[0]/img.shape[1], min_y=box[1]/img.shape[0], max_x=box[2]/img.shape[1], max_y=box[3]/img.shape[0], confidence=score)
        result.append(b)
    return result


    
    #bounding_boxes: List[BoundingBoxClassification] = []
    #for _ in range(random.randint(0, 9)):
    #    bounding_box: BoundingBoxClassification = get_dummy_box()
    #    bounding_boxes.append(bounding_box)
    #    logger.info(bounding_box)
    #return bounding_boxes


def get_dummy_box() -> BoundingBoxClassification:
    random_class = random.randint(0, 1)  # 0 = PIG, 1 = PIGLET
    random_min_x = random.uniform(0, .9)
    random_min_y = random.uniform(0, .9)
    random_max_x = random.uniform(random_min_x + .05, 1)
    random_max_y = random.uniform(random_min_y + .05, 1)
    return BoundingBoxClassification(
        class_id=random_class,
        min_x=random_min_x,
        min_y=random_min_y,
        max_x=random_max_x,
        max_y=random_max_y,
        confidence=random.uniform(0, 1)
    )
