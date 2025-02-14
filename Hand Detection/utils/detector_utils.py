import numpy as np
import sys
import tensorflow as tf
import os
from threading import Thread
from datetime import datetime
import cv2
from utils import label_map_util
from collections import defaultdict
from keras.preprocessing import image
from keras.utils import to_categorical
from ibm_watson import VisualRecognitionV3

detection_graph = tf.Graph()
sys.path.append("..")

_score_thresh = 0.20

MODEL_NAME = 'hand_inference_graph'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join(MODEL_NAME, 'hand_label_map.pbtxt')

NUM_CLASSES = 1
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class WebcamVideoStream:
    def __init__(self, src, width, height):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def size(self):
        return self.stream.get(3), self.stream.get(4)

    def stop(self):
        self.stopped = True

def load_inference_graph():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    return detection_graph, sess

def detect_objects(image_np, detection_graph, sess):
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)

#######################################################################################
#Drawing green box over hand

def draw_box_on_image(num_hands_detect, score_thresh, scores, boxes, im_width, im_height, image_np):
    for i in range(num_hands_detect):
        if (scores[i] > score_thresh):
            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width, boxes[i][0] * im_height, boxes[i][2] * im_height)
            
            left = int(left) - 30
            right = int(right) + 30
            top = int(top) - 10
            bottom = int(bottom) + 10

            p1 = (int(left), int(top))
            width = int(right) - int(left)
            height = int(bottom) - int(top)

            if width > height:
                p2 = (int(left) + width, int(top) + width)
                new_image_np = image_np[int(top):int(top) + width, int(left):int(left) + width]
            else:
                p2 = (int(left) + height, int(top) + height)
                new_image_np = image_np[int(top):int(top) + height, int(left):int(left) + height]

            cv2.rectangle(image_np, p1, p2, (77, 255, 9), 3, 1)
            new_image_np = cv2.resize(new_image_np, (200,200))
            new_image_np = cv2.cvtColor(new_image_np, cv2.COLOR_BGR2RGB)
            return new_image_np

#######################################################################################
#prediction

def predict(flag, path):
    if(flag == 1):
        #IBMCLOUD2
        api_key = 'ViuVYAOjt57hl614G9SqWBeRP83TNB8-UhtP15mH6vMv'
        version = '2018-03-19'
        model = 'a2n_2139996789'
    elif(flag == 2):
        #IBMCLOUD3
        api_key = 'EaLMuHXAN2-98h5ubcIif392qAWXUpy3cMpnrRXBdFgL'
        version = '2018-03-19'
        model = 'h_n_model_1566440431'
    
    visual_recognition = VisualRecognitionV3(version,iam_apikey=api_key)

    with open(path, 'rb') as images_file:
        classes = visual_recognition.classify(images_file,threshold='0.6',classifier_ids=model).get_result()
    
    var=classes["images"][0]["classifiers"][0]["classes"][0]["class"]
    if(var=="SPACE"):
        return " "
    return var

#######################################################################################
#prediction

def Predict(classifier, img):
    img = cv2.resize(img, (50,50))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    result_c = classifier.predict(img)
    result = np.argmax(result_c, axis = 1) 

    if(result==[0]):
        print("A")
    elif(result==[1]):
        print("B")
    elif(result==[2]):
        print("C")
    elif(result==[3]):
        print("D")
    elif(result==[4]):
        print("E")
    elif(result==[5]):
        print("F")
    elif(result==[6]):
        print("G")
    elif(result==[7]):
        print("H")
    elif(result==[8]):
        print("I")
    elif(result==[9]):
        print("J")
    elif(result==[10]):
        print("K")
    elif(result==[11]):
        print("L")
    elif(result==[12]):
        print("M")
    elif(result==[13]):
        print("N")
    elif(result==[14]):
        print("O")
    elif(result==[15]):
        print("P")
    elif(result==[16]):
        print("Q")
    elif(result==[17]):
        print("R")
    elif(result==[18]):
        print("S")
    elif(result==[19]):
        print("T")
    elif(result==[20]):
        print("U")
    elif(result==[21]):
        print("V")
    elif(result==[22]):
        print("W")
    elif(result==[23]):
        print("X")
    elif(result==[24]):
        print("Y")
    elif(result==[25]):
        print("Z")
    elif(result==[26]):
        print(" ")

    return result;