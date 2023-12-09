import tensorflow as tf
import os
from helper import *

# TensorFlow 그래프와 세션을 전역으로 로드
graph_def = tf.compat.v1.GraphDef()
labels = []

# 모델과 레이블 파일 로드
filename = "model.pb"
labels_filename = "labels.txt"

with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

# 전역 TensorFlow 세션 생성
sess = tf.compat.v1.Session()

def is_front(imageFile):
    image = Image.open(imageFile)
    image = update_orientation(image)
    image = convert_to_opencv(image)
    image = resize_down_to_1600_max_dim(image)

    h, w = image.shape[:2]
    min_dim = min(w, h)
    max_square_image = crop_center(image, min_dim, min_dim)
    augmented_image = resize_to_256_square(max_square_image)

    input_tensor_shape = sess.graph.get_tensor_by_name('Placeholder:0').shape.as_list()
    network_input_size = input_tensor_shape[1]
    augmented_image = crop_center(augmented_image, network_input_size, network_input_size)

    output_layer = 'loss:0'
    input_node = 'Placeholder:0'

    try:
        prob_tensor = sess.graph.get_tensor_by_name(output_layer)
        predictions = sess.run(prob_tensor, {input_node: [augmented_image] })
    except KeyError:
        print ("Couldn't find classification output layer: " + output_layer + ".")
        print ("Verify this a model exported from an Object Detection project.")
        exit(-1)

    highest_probability_index = np.argmax(predictions)
    return labels[highest_probability_index]
