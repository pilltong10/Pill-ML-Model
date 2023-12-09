# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
"""Sample prediction script for TensorFlow 2.x."""
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection

import os
import concurrent.futures

MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float64)[:, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]

def crop_detected_object(image, detection, margin_ratio=0.1):
    width, height = image.size
    box = detection['boundingBox']
    left = max(int(box['left'] * width - margin_ratio * width), 0)
    top = max(int(box['top'] * height - margin_ratio * height), 0)
    right = min(int((box['left'] + box['width']) * width + margin_ratio * width), width)
    bottom = min(int((box['top'] + box['height']) * height + margin_ratio * height), height)

    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def process_directory(input_dir, output_dir, od_model, margin_ratio=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for image_filename in os.listdir(input_dir):
            if image_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, image_filename)
                output_path = os.path.join(output_dir, os.path.basename(image_filename))
                futures.append(executor.submit(process_image, image_path, output_path, od_model, margin_ratio=0.05))

        for future in concurrent.futures.as_completed(futures):
            future.result()

def process_image(image_path, output_path, od_model, margin_ratio=0):
    image = Image.open(image_path)
    predictions = od_model.predict_image(image)

    if predictions:  # Check if there are any detections
        cropped_object = crop_detected_object(image, predictions[0], margin_ratio)
        cropped_object.save(output_path)
        print(f"Processed and saved: {output_path}")


def main(input_base_dir, output_base_dir):
    # Load a TensorFlow model
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())

    # Load labels
    with open(LABELS_FILENAME, 'r') as f:
        labels = [label.strip() for label in f.readlines()]

    od_model = TFObjectDetection(graph_def, labels)

    # Process each subdirectory in the input directory
    for root, dirs, files in os.walk(input_base_dir):
        for dir in dirs:
            input_dir = os.path.join(root, dir)
            output_dir = os.path.join(output_base_dir, dir + "_cropped")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            process_directory(input_dir, output_dir, od_model)


if __name__ == '__main__':
    # if len(sys.argv) != 3:
    #     print("Usage: python predict.py <input_dir> <output_dir>")
    #     sys.exit(-1)

    # input_dir = sys.argv[1]
    # output_dir = sys.argv[2]

    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    input_dir = r'C:\Users\xerop\Desktop\pill_classification\테스트이미지_스마트폰_원본'
    output_dir = r'C:\Users\xerop\Desktop\pill_classification\테스트이미지_스마트폰_cropped'

    main(input_dir, output_dir)
