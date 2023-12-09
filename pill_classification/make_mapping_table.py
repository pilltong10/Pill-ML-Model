import json
import os
from concurrent.futures import ThreadPoolExecutor

# 디렉토리 경로
source_directory = r'C:\Users\xerop\Desktop\AI 모델 소스코드\평가용 데이터셋\pill_data\pill_data_croped'

# 최종 결과를 저장할 리스트
result = {"images": []}

def process_directory(directory):
    dir_path = os.path.join(source_directory, directory)
    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(dir_path, filename)
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                image_data = {
                    "drug_N": data["images"][0]["drug_N"],
                    "dl_name": data["images"][0]["dl_name"],
                    "img_key": data["images"][0]["img_key"],
                    "dl_material": data["images"][0]["dl_material"],
                    "di_class_no": data["images"][0]["di_class_no"],
                    "chart": data["images"][0]["chart"]
                }
                result["images"].append(image_data)
        break
    
with ThreadPoolExecutor() as executor:
    # 각 디렉토리를 병렬로 처리
    for root, dirs, files in os.walk(source_directory):
        for directory in dirs:
            executor.submit(process_directory, directory)

# 결과를 JSON 파일로 저장
output_file = 'integrated_data.json'
with open(output_file, 'w', encoding='utf-8') as output_json:
    json.dump(result, output_json, ensure_ascii=False, indent=4)
