import cv2
import tensorflow as tf
import datetime
import argparse
from utils import detector_utils as detector_utils
from keras.models import load_model
from imgaug import augmenters as iaa

detection_graph, sess = detector_utils.load_inference_graph()

#Test_set
saveMode=False;
count=0
rotate=iaa.Affine(rotate=(-25, 25))
noise=iaa.AdditiveGaussianNoise(scale=(10, 40))
crop= iaa.Crop(percent=(0, 0.2))
shear=iaa.Affine(shear=(-16, 16))
sharpen=iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))

mulitple_augment=iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-30, 30), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255) # if mode is constant, use a cval between 0 and 25
        )
path = 'A2E_validation/J/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-sth', '--scorethreshold', dest='score_thresh', type=float, default=0.4, help='Score threshold for displaying bounding boxes')
    parser.add_argument('-fps', '--fps', dest='fps', type=int, default=1, help='Show FPS on detection/display visualization')
    parser.add_argument('-src', '--source', dest='video_source', default=0, help='Device index of the camera.')
    parser.add_argument('-wd', '--width', dest='width', type=int, default=480, help='Width of the frames in the video stream.')
    parser.add_argument('-ht', '--height', dest='height', type=int, default=640, help='Height of the frames in the video stream.')
    parser.add_argument('-ds', '--display', dest='display', type=int, default=0, help='Display the detected images using OpenCV. This reduces FPS')
    args = parser.parse_args()

    classifier = load_model('trained_models/shuv_2e.h5')

    cap = cv2.VideoCapture(0)
    
    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    num_hands_detect = 1

    while True:
        try:   
            ret, image_np = cap.read()
            image_np = cv2.flip(image_np, 1) 
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
   
            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
            new_image_np = detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh, scores, boxes, im_width, im_height, image_np)

            num_frames += 1
            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            fps = num_frames / elapsed_time

            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            cv2.imshow('Single-Threaded Detection', image_np)

            k=cv2.waitKey(10) & 0xFF
            if k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
            elif k == ord('p'):
                detector_utils.keras_process_predict(classifier, new_image_np)
            
            elif k==ord('w'):
                print("clicked on w")
                cv2.imwrite('tempWatson.jpg',new_image_np)
                detector_utils.watson_predict('tempWatson.jpg')
                print("finished clicked on w")
                
            elif k == ord('s'):
                image = new_image_np

                count+=1
                cv2.imwrite(path + 'img_'+str(count)+'.jpg',image)

                image_aug= rotate.augment_image(image)
                cv2.imwrite(path + 'img_'+str(count)+'_r.jpg',image_aug)
                
                image_aug= noise.augment_image(image)
                cv2.imwrite(path + 'img_'+str(count)+'_n.jpg',image_aug)
            
                image_aug= shear.augment_image(image)
                cv2.imwrite(path + 'img_'+str(count)+'_sr.jpg',image_aug)
                
                image_aug= sharpen.augment_image(image)
                cv2.imwrite(path + 'img_'+str(count)+'_sp.jpg',image_aug)
           
                print("clicked correctly")


            elif k == ord('t'):
                image = new_image_np

                image_aug= rotate.augment_image(image)
                cv2.imshow("rotate",image_aug)
                image_aug= noise.augment_image(image)
                cv2.imshow("noise",image_aug)
                image_aug= shear.augment_image(image)
                cv2.imshow("shear",image_aug)
                image_aug= sharpen.augment_image(image)
                cv2.imshow("sharpen",image_aug)
            try:
                cv2.imshow("Mask", new_image_np)
            except:
                cv2.destroyWindow("Mask")
                continue;
        except:
            continue