import cv2
import sys
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scrfd_ import SCRFD
sys.path.append("/home/tima/detec_and_tracking/deep_sort/")
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from tracking_helpers import read_class_names, create_box_encoder
import pandas as pd

def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)

def track_database(dataframe = '/home/tima/detec_and_tracking/data.csv'):
    
    data = pd.read_csv(dataframe)

    idx = data['id'].values
    feature_face_values = data['feature_face'].values
    feature_mask_values = data['feature_mask'].values

    for i in range(len(idx)):
        feature_face_values_str = str(feature_face_values[i])  # Convert float to string
        feature_face_values_str = feature_face_values_str.replace('[', '').replace(']', '')
        feature_face_values_str = feature_face_values_str.replace('\n', '')
        feature_face_values[i] = np.fromstring(feature_face_values_str, sep=' ')

        feature_mask_values_str = str(feature_mask_values[i])  # Convert float to string
        feature_mask_values_str = feature_mask_values_str.replace('[', '').replace(']', '')
        feature_mask_values_str = feature_mask_values_str.replace('\n', '')
        feature_mask_values[i] = np.fromstring(feature_mask_values_str, sep=' ')
    
    return idx, feature_face_values, feature_mask_values


class SCRFD_DeepSort:
    
    def __init__(self, model_tracking_path: str, model_mask, detect, detector, max_cosine_distance:float=0.2, nn_budget:float=None, nms_max_overlap:float=1):
        self.detector = detector
        self.nms_max_overlap = nms_max_overlap
        self.encoder = create_box_encoder(model_tracking_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)
        self.class_names = read_class_names()
        self.model_mask = model_mask
        self.detect = detect
  
    def track_video(self, video:str, output:str, skip_frames:int=0, show_live:bool=False, count_objects:bool=False, verbose:int = 0, window_name:str=None):
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)
        
        # out = None
        # if output:
            # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
            # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # fps = int(vid.get(cv2.CAP_PROP_FPS))
            # codec = cv2.VideoWriter_fourcc(*"XVID")
            # out = cv2.VideoWriter(output, codec, fps, (width, height))
            
        frame_num = 0
        f2 = []
        check_features = []
        name = []
        count_check = 0
        start_frame_time = time.time()
        # with concurrent.futures.ThreadPoolExecutor() as executor:
        
        idx, feature_face_values, feature_mask_values = track_database()
                   
        while True:
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed')
                break
            frame_num += 1
            
            # if skip_fram es and not frame_num % skip_frames: continue
            if skip_frames > 0 and frame_num % skip_frames != 0: continue

            if verbose >=1: start_time = time.time()
            
            elapsed_time = time.time() - start_frame_time

            if elapsed_time > 0.05:
                # print("reset")
                start_frame_time = time.time()
                continue
            # print(elapsed_time)
            # # future = executor.submit(self.detector.detect, frame, 0.7)
            # # dets, kps = future.result()
            
            dets, _ = self.detector.detect(frame, 0.6)
            
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
                
            else:   
                bboxes = dets[:,:4]
                box = np.copy(bboxes)
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh

                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
                
                scores = dets[:,4]
                classes = dets[:,-1]
                num_objects = bboxes.shape[0]
            
            names = []
            
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)
            
            names = np.array(names)
            count = len(names)
            
            if count_objects:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

        # ---------------------------------- DeepSORT tracker work starts here ------------------------------------------------------------
        
            # ---------------------------------- Check mask + id ----------------------------------
            features = self.encoder(frame, bboxes)
            if len(features) > 0:
                if len(f2) > 0:
                    check_features = [dot_product(features[i], f2[i]) for i in range(min(len(features), len(f2)))]
                    
                f2 = features
                
                if len(check_features) == 0:
                    print("Start")
                    for i in range(len(box)):
                        label = self.detect.mask_image(frame, self.model_mask, [box[i]])
                        if label:
                            for j, feature_value in enumerate(feature_mask_values):
                                check = dot_product(features[0], feature_value)
                                if check > 0.9:
                                    name.append(idx[j])
                                    count_check = 0
                                    break   
                        else:
                            for j, feature_value in enumerate(feature_face_values):
                                check = dot_product(features[i], feature_value)
                                if check > 0.9 :
                                    name.append(str(idx[j]))
                                    count_check = 0
                                    break
        
                else:
                    for check_feature in check_features:
                        if check_feature < 0.8:
                            check_feature = []
                            name = []
                            f2 = []
                            break
                        
                if len(name)==0 and count_check<24:
                    check_feature = []
                    f2 = []
                    print("Reset")
                    count_check += 1
                    
            else:
                name = []
                check_features = []  
                f2 = []
            
            
            if len(name)!=0:
                names = name
            
            print(name)
            
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices] 
              
    
            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain
                
            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                bbox = track.to_tlbr()
                class_name = track.get_class()

                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
    
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name ,(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    

            #     if verbose == 2:
            #         print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if verbose >= 1:
                print((time.time() - start_time))
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
            
            result = np.asarray(frame)
            # result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
                    
            # if output: out.write(result) # save output video
    
            if show_live:
                cv2.imshow(window_name, result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
    
        
        cv2.destroyAllWindows()


