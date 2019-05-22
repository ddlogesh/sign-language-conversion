from utils import detector_utils as detector_utils
import cv2
import tensorflow as tf
import datetime
import argparse
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

classifier = load_model('a2z.h6')
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def keras_process_predict(img):
    img = cv2.resize(img, (200,200))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = np.vstack([img])
    result = classifier.predict_classes(img)
    print (result)
    
detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.4, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int, default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source', default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int, default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=640, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int, default=1, help='Display the detected images using OpenCV. This reduces FPS')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    num_hands_detect = 1

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    while True:
        ret, image_np = cap.read()
        img = cv2.flip(image_np, 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        try:
            img = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        brown = cv2.inRange(hsv,np.array([2, 50, 60]), np.array([25, 150, 255]))
	
        kernal = np.ones((5, 5), "uint8")
        brown = cv2.dilate(brown, kernal)
        res_brown = cv2.bitwise_and(img, img, mask = brown)
        (contours,hierarchy)=cv2.findContours(brown, cv2.RETR_TREE, cv2)
        for pic, contour in enumerate(contours):
            try:
                area = cv2.contourArea(contour)
           
                x, y, w, h = cv2.boundingRect(contour)
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                
                brown = cv2.cvtColor(brown,cv2.COLOR_GRAY2RGB)
                cv2.drawContours(brown, contours, -1, (0,255,0), 1)
                brown = brown[y:y+h, x:x+w]
                brown = cv2.resize(brown, (350,350))
            except:
                continue;
        boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time    
                                               
        if (args.display > 0):
            cv2.imshow('Single-Threaded Detection', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imshow('Mask', brown)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            elif k==ord('p'):
            	keras_process_predict(image)
      
    