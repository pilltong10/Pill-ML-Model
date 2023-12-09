import json
import pill_getname
import os
import glob
from msrest.authentication import ApiKeyCredentials
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient


ENDPOINT = 'https://southcentralus.api.cognitive.microsoft.com/'
prediction_key = '8724912124724c6d9c18b7202e80d8ed'

# 이미지 경로
base_image_location = r'C:\Users\xerop\Desktop\pill_classification\테스트이미지_스마트폰_원본\게보린'
project_id = '410b4af7-d605-4054-b1a1-0d8b6a26cec3'
iteration_name = 'Iteration3' # 나중에 이 부분을 더 나은 ITERATION에 맞게 수정하면 됨.
print("running....")

if __name__ == '__main__':
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
    print("running....")
    # 디렉토리 내의 모든 PNG 이미지에 대해 반복
    for image_file in glob.glob(os.path.join(base_image_location, '*.jpg')):
        with open(image_file, "rb") as image_contents:
            results = predictor.classify_image(
                project_id, iteration_name, image_contents.read())
            print(image_file + " : ")
            for i, prediction in enumerate(results.predictions):
                pill_name = pill_getname.pill_getname(prediction.tag_name)
                probability = prediction.probability * 100
                print(f"{pill_name}: {probability:.2f}%")
                if i >= 4: break


