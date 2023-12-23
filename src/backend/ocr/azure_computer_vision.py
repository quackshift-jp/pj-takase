import os
from dotenv import load_dotenv
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from msrest.pipeline import ClientRawResponse
from azure.cognitiveservices.vision.computervision.models import (
    ComputerVisionOcrErrorException,
    OperationStatusCodes,
)
import time


load_dotenv(verbose=True)
KEY = os.environ.get("AZURE_COMPUTER_VISION_API_KEY_1")
ENDPOINT = os.environ.get("AZURE_COMPUTER_VISION_ENDPOINT")

CLIENT = ComputerVisionClient(ENDPOINT, CognitiveServicesCredentials(KEY))


def read_image_response(image_path: str) -> ClientRawResponse:
    try:
        with open(image_path, "rb") as image_file:
            response = CLIENT.read_in_stream(image_file, language="ja", raw=True)
            return response
    except ComputerVisionOcrErrorException as e:
        print("errors:", e.response)
        raise e


def get_operation_id(image_path: str) -> str:
    response = read_image_response(image_path)
    operation_location = response.headers["Operation-Location"]
    return operation_location.split("/")[-1]


def extract_text(operation_id: str) -> str:
    extracted_text = []
    while True:
        result = CLIENT.get_read_result(operation_id)
        if result.status not in ["notStarted", "running"]:
            break
        time.sleep(1)

    if result.status == OperationStatusCodes.succeeded:
        for line in result.analyze_result.read_results[0].lines:
            extracted_text.append(line.text)
    return "".join(extracted_text)
