import os
import shutil
import random
import concurrent.futures
import classifier_v2

# 원본 이미지 디렉토리
source_directory = r'C:\Users\xerop\Desktop\AI 모델 소스코드\평가용 데이터셋\pill_data\pill_data_croped'

# 대상 디렉토리
target_base_directory = r'C:\Users\xerop\Desktop\알약이미지\학습이미지'

image_number = 200
directory_number = 100
finished1 = 0
finished2 = 0

def copy_images(source_path, target_directory):
    global finished1
    global finished2
    # 이미지 파일 찾기
    image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 무작위로 이미지 선택 및 검사
    random.shuffle(image_files)
    selected_images = []
    for image in image_files:
        if len(selected_images) >= image_number:
            break
        image_path = os.path.join(source_path, image)

        is_front = classifier_v2.is_front(image_path)
        print("runnung")
        print("source_path-" + str(source_path) + "finished1 : " + str(finished1) + "/20000"+ "finished2 : " + str(finished2) + "/20000")
        
        if is_front == 'front':
            selected_images.append(image)
            finished1 = finished1 + 1

    # 이미지를 대상 디렉토리로 복사
    for image in selected_images:
        source_image_path = os.path.join(source_path, image)
        target_image_path = os.path.join(target_directory, image)
        shutil.copy(source_image_path, target_image_path)

        # print(f"복사 완료: {source_image_path} -> {target_image_path}")

        finished2 = finished2 + 1
        # print("완료된 작업의 수 : " + str(finished))

# 모든 하위 디렉토리 목록
all_subdirectories = [os.path.join(root, dir) for root, dirs, _ in os.walk(source_directory) for dir in dirs]

# 랜덤하게 100개의 디렉토리 선택
selected_random_directories = random.sample(all_subdirectories, directory_number)

# 특정 디렉토리 추가
specific_directories = ['K-000573', 'K-044382', 'K-010158', 'K-013395', 'K-007060']
selected_directories = selected_random_directories + [os.path.join(source_directory, dir) for dir in specific_directories if os.path.join(source_directory, dir) in all_subdirectories]

# 각 디렉토리에 대해 병렬로 작업 수행
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
    for source_path in selected_directories:
        target_directory = os.path.join(target_base_directory, os.path.basename(source_path))

        # 대상 디렉토리가 없으면 생성
        if not os.path.exists(target_directory):
            os.makedirs(target_directory)

        executor.submit(copy_images, source_path, target_directory)
