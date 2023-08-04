import time
import threading
import cv2
import PIL.Image
import PIL.ImageTk
import sys
from numpy.linalg import norm
from scrfd_ import SCRFD
sys.path.append("/home/tima/detec_and_tracking/deep_sort/")
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker

from tracking_helpers import read_class_names, create_box_encoder
import pandas as pd

from scrfd_ import SCRFD
from scrfd_deepsort import *

sys.path.append('/home/tima/detec_and_tracking/Face-Mask-Detection')
from detect_mask_image import detect_mask_image, load_model_detect_mask

model_mask =load_model_detect_mask()
detect = detect_mask_image(model=model_mask)

detector = SCRFD(model_file='/home/tima/detec_and_tracking/insightface/detection/scrfd/onnx/scrfd_500m.onnx')
detector.prepare(-1)
encoder = create_box_encoder(model_filename="/home/tima/detec_and_tracking/deep_sort/deep_sort/model_weights/mars-small128.pb", batch_size=1)
from elasticsearch import Elasticsearch, helpers

es = Elasticsearch("http://0.0.0.0:9200/")

def _check_exits_customer_id(elastic_search, vector):
        query = {
            "min_score": 0.92,
            "query": {
                "function_score": {
                    "boost_mode": "replace",
                    "script_score": {
                        "script": {
                            "source": "binary_vector_score",
                            "lang": "knn",
                            "params": {
                                "cosine": True,
                                "field": "face_embedding",
                                "vector": vector
                            }
                        }
                    }
                }
            },
            "_source": 'name'
        }
        res = elastic_search.search(index='id_face', body=query)
        if res['hits']['total']['value'] > 0:
            return res['hits']['hits'][0]['_source']['name']
        else: return None
        
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

class VideoCapture(threading.Thread):

    def __init__(self, video_source=0, width=None, height=None, fps=None):
        """TODO: add docstring"""
        threading.Thread.__init__(self)
        self.video_source = video_source
        self.width = width
        self.height = height
        self.fps = fps

        self.running = False

        # Open the video source
        # print("gia tri",video_source)
        self.vid = cv2.VideoCapture(video_source)
        
        if not self.vid.isOpened():
            raise ValueError("[MyVideoCapture] Unable to open video source", video_source)

        # Get video source width and height
        if not self.width:
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))    # convert float to int
        if not self.height:
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))  # convert float to int
        if not self.fps:
            self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))              # convert float to int

        # default value at start
        self.ret = False
        self.frame = None

        self.convert_color = cv2.COLOR_BGR2RGB
        #self.convert_color = cv2.COLOR_BGR2GRAY
        self.convert_pillow = True

        # default values for recording
        self.recording = False
        self.recording_filename = 'output.mp4'
        self.recording_writer = None

        # start thread
        self.running = True
        self.thread = threading.Thread(target=self.process)
        self.thread.start()

    def snapshot(self, filename=None):
        """TODO: add docstring"""

        if not self.ret:
            print('[MyVideoCapture] no frame for snapshot')
        else:
            if not filename:
                filename = time.strftime("frame-%d-%m-%Y-%H-%M-%S.jpg")

            if not self.convert_pillow:
                cv2.imwrite(filename, self.frame)
                print('[MyVideoCapture] snapshot (using cv2):', filename)
            else:
                self.frame.save(filename)
                print('[MyVideoCapture] snapshot (using pillow):', filename)

    def start_recording(self, filename=None):
        """TODO: add docstring"""

        if self.recording:
            print('[MyVideoCapture] already recording:', self.recording_filename)
        else:
            # VideoWriter constructors
            #.mp4 = codec id 2
            if filename:
                self.recording_filename = filename
            else:
                self.recording_filename = time.strftime("%Y.%m.%d %H.%M.%S", time.localtime()) + ".avi"
            #fourcc = cv2.VideoWriter_fourcc(*'I420') # .avi
            #fourcc = cv2.VideoWriter_fourcc(*'MP4V') # .avi
            fourcc = cv2.VideoWriter_fourcc(*'MP42') # .avi



            self.recording_writer = cv2.VideoWriter(self.recording_filename, fourcc, self.fps, (self.width, self.height))
            self.recording = True
            print('[MyVideoCapture] started recording:', self.recording_filename)

    def stop_recording(self):
        """TODO: add docstring"""

        if not self.recording:
            print('[MyVideoCapture] not recording')
        else:
            self.recording = False
            self.recording_writer.release()
            print('[MyVideoCapture] stop recording:', self.recording_filename)

    def record(self, frame):
        """TODO: add docstring"""

        # write frame to file
        if self.recording_writer and self.recording_writer.isOpened():
            self.recording_writer.write(frame)
            
    
    
    def process(self):
        """TODO: add docstring"""
        
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)

        class_names = read_class_names()
        tracker = Tracker(metric)
    
        frame_num = 0
        name = []
        count_objects = True
        start_frame_time = time.time()
        dets = None
        features = []
        f = 0
        t = 0
        while self.running:
            ret, frame = self.vid.read()
            if ret:
                frame_num += 1
                start_time = time.time()
            
                elapsed_time = time.time() - start_frame_time

                if elapsed_time > 0.1:
                    start_frame_time = time.time()
                    continue
                
                t1 = time.time()
                if frame_num == 1 or frame_num % 10 == 0:
                    det, _ = detector.detect(frame, 0.6)
                    dets = np.copy(det)
                    
                print("Thoi gian detect", round((time.time()-t1),2))
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if dets is None:
                    bboxes = []
                    scores = []
                    classes = []
                    num_objects = 0
                    
                else:   
                    bboxes = np.copy(dets[:,:4])
                    bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh

                    bboxes[:,3] = bboxes[:,3] - bboxes[:,1]
                    
                    scores = dets[:,4]
                    classes = dets[:,-1]
                    num_objects = bboxes.shape[0]
                
                names = []
                
                for i in range(num_objects):
                    class_indx = int(classes[i])
                    class_name = class_names[class_indx]
                    names.append(class_name)
                
                names = np.array(names)
                count = len(names)

                if count_objects:
                    cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)

            # ---------------------------------- DeepSORT tracker work starts here ------------------------------------------------------------
            
                # ---------------------------------- Check mask + id ----------------------------------
                
                if frame_num % 15 == 0:
                    t2 = time.time()
                    feature = encoder(frame, bboxes)
                    print("Thoi gian encode", time.time()-t2)
                    features = np.copy(feature)
                    name = []
                    
                    if len(features) > 0:
                        t3 = time.time()
                        for feat in features:
                            n = _check_exits_customer_id(es, (feat/norm(feat)).tolist())
                            if n is not None:
                                name.append(n)
                        print("Thoi gian tim kiem", time.time()-t3)
   
                if len(name)!=0:
                    names = name
                
                print(names)
                
                t2 = time.time()
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

                cmap = plt.get_cmap('tab20b') #initialize color map
                colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

                boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
                scores = np.array([d.confidence for d in detections])
                classes = np.array([d.class_name for d in detections])
                indices = preprocessing.non_max_suppression(boxs, classes, 1, scores)
                detections = [detections[i] for i in indices] 
                
        
                tracker.predict()  # Call the tracker
                tracker.update(detections) #  updtate using Kalman Gain
                print("---------------------------", time.time() - t2)
                for track in tracker.tracks:  # update new findings AKA tracks
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                    color = [i * 255 for i in color]
        
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name ,(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                    
            #     #     if verbose == 2:
            #     #         print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                        
            #     # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
               
            #     print((time.time() - start_time))
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: ")

                
                frame = cv2.resize(frame, (self.width, self.height))
                cv2.putText(frame, "FPS: {}".format(round(fps, 2)), (frame.shape[1]-105, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8 , (0, 30, 30), 2)  

                # it has to record before converting colors
                if self.recording:
                    self.record(frame)

                if self.convert_pillow:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = PIL.Image.fromarray(frame)
            else:
                print('[MyVideoCapture] stream end:', self.video_source)
                # TODO: reopen stream
                self.running = False
                if self.recording:
                    self.stop_recording()
                break

            # assign new frame
            self.ret = ret
            self.frame = frame

            # sleep for next frame
            time.sleep(1/self.fps)

    def get_frame(self):
        """TODO: add docstring"""

        return self.ret, self.frame

    # Release the video source when the object is destroyed
    def __del__(self):
        """TODO: add docstring"""

        # stop thread
        if self.running:
            self.running = False
            self.thread.join()

        # relase stream
        if self.vid.isOpened():
            self.vid.release()
