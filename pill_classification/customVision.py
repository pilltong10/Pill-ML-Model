from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid

# retrieve environment variables
ENDPOINT = os.environ["VISION_TRAINING_ENDPOINT"]
training_key = os.environ["VISION_TRAINING_KEY"]
prediction_key = os.environ["VISION_PREDICTION_KEY"]
prediction_resource_id = os.environ["VISION_PREDICTION_RESOURCE_ID"]

if __name__ == '__main__':
    credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
    trainer = CustomVisionTrainingClient(ENDPOINT, credentials)
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    publish_iteration_name = "classifyModel"

    credentials = ApiKeyCredentials(in_headers={"Training-key": training_key})
    trainer = CustomVisionTrainingClient(ENDPOINT, credentials)

    # Create a new project
    print ("Creating project...")
    project_name = uuid.uuid4()
    project = trainer.create_project(project_name)

    base_image_location = os.path.join (os.path.dirname(__file__), "Images")

    print("Adding images...")


    source_directory = r'C:\Users\xerop\Desktop\알약이미지\학습이미지'
    # C:\Users\xerop\Desktop\AI 모델 소스코드\평가용 데이터셋\pill_data\pill_data_croped

    num_of_dirs = 105 # 업로드할 알약의 종류
    num_of_img = 200 # 알약 당 이미지 개수
    #test_img_list = ['K-000573', 'K-044382', 'K-010158', 'K-013395', 'K-007060'] # 테스트할 실제 이미지가 있는 알약 리스트

    for root, dirs, files in os.walk(source_directory):

            for i, directory in enumerate(dirs):

                image_list = []

                # if(directory in test_img_list): continue # test이미지는 추가했으므로 넘김                
                # if(i < len(test_img_list)):
                #     directory = test_img_list[i] # test이미지 추가
                # if i > num_of_dirs:
                #     break  # 정해진 개수만큼만

                source_path = os.path.join(root, directory)

                image_files = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                tag = trainer.create_tag(project.id, directory)

                for i, image in enumerate(image_files):
                    if i > num_of_img:
                        break
                    image_path = os.path.join(source_path, image)
                    with open(image_path, "rb") as image_contents:
                        image_list.append(ImageFileCreateEntry(name=image, contents=image_contents.read(), tag_ids=[tag.id]))
                    if(i != 0 and (i % 64 == 63 or i == num_of_img - 1)):
                        upload_result = trainer.create_images_from_files(project.id, ImageFileCreateBatch(images=image_list))
                        del image_list[:]   # 배열 초기화(한 번에 64개 업로드 한계가 있음)

                        if not upload_result.is_batch_successful:
                            print("Image batch upload failed.")
                            print(upload_result)
                            for image in upload_result.images:
                                print("Image status: ", image.status)
                        else:
                            print(f"{directory}: OK {i + 1}/{num_of_img}")

    # print ("Training...")
    # iteration = trainer.train_project(project.id)
    # while (iteration.status != "Completed"):
    #     iteration = trainer.get_iteration(project.id, iteration.id)
    #     print ("Training status: " + iteration.status)
    #     print ("Waiting 10 seconds...")
    #     time.sleep(10)
                    
    # # The iteration is now trained. Publish it to the project endpoint
    # trainer.publish_iteration(project.id, iteration.id, publish_iteration_name, prediction_resource_id)
    print ("Done!")



