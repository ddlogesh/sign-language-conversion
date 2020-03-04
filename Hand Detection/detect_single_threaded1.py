import cv2
import numpy as np
import tensorflow as tf
import argparse
from utils import detector_utils as detector_utils
from keras.models import load_model
from imgaug import augmenters as iaa
import time
#import tensorflow.python.util.deprecation as deprecation

#deprecation._PRINT_DEPRECATION_WARNINGS = False
detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.4, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int, default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source', default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int, default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=640, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int, default=0, help='Display the detected images using OpenCV. This reduces FPS')
    args = parser.parse_args()

    cap = cv2.VideoCapture(0)
    
    im_width, im_height = (cap.get(3), cap.get(4))
    num_hands_detect = 1
    pred_img = np.zeros((500,500,3), np.uint8)
    pred = ""


    while True:
        try:
            ret, image_np = cap.read()
            image_np = cv2.flip(image_np, 1) 
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)   
        
            cv2.imshow('Single-Threaded Detection', image_np)
            

            start = time.time() 
            while time.time() - start < 2:
                pass 


            k=cv2.waitKey(10) & 0xFF

            if k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            elif True:
                boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
                new_image_np = detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite('sample.jpg', new_image_np)
                #pred += detector_utils.predict(1,'sample.jpg')
                pred='a'
                cv2.putText(pred_img, pred, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                cv2.imshow('Prediction', pred_img)

            try:
                cv2.imshow("Mask", new_image_np)
            except:
                cv2.destroyWindow("Mask")
                continue;
        except:
            continue

#Akilogesh      [ABCDEFGIN ]
#Ddlogesh555    [HKLMOPRSU]
