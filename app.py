#!/usr/bin/env python

# author: Bartlomiej "furas" Burek (https://blog.furas.pl)
# date: 2021.01.26

import time
import threading
import cv2
import PIL.Image
import PIL.ImageTk
import tkinter
import tkinter.filedialog
import os
import sys

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


"""TODO: add docstring"""
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
            #fourcc = cv2.VideoWriter_fourcc(*'AVC1') # error libx264
            #fourcc = cv2.VideoWriter_fourcc(*'H264') # error libx264
            #fourcc = cv2.VideoWriter_fourcc(*'WRAW') # error --- no information ---
            #fourcc = cv2.VideoWriter_fourcc(*'MPEG') # .avi 30fps
            #fourcc = cv2.VideoWriter_fourcc(*'MJPG') # .avi
            #fourcc = cv2.VideoWriter_fourcc(*'XVID') # .avi
            #fourcc = cv2.VideoWriter_fourcc(*'H265') # error


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
        f2 = []
        check_features = []
        name = []
        count_check = 0
        count_objects = True
        start_frame_time = time.time()
        start_fps = time.time()
        idx, feature_face_values, feature_mask_values = track_database()
        dets = None
        features = []
        
        while self.running:
            ret, frame = self.vid.read()
            if ret:
                frame_num += 1
                start_time = time.time()
            
                elapsed_time = time.time() - start_frame_time

                if elapsed_time > 0.08:
                    # print("reset")
                    start_frame_time = time.time()
                    continue
                
                if frame_num == 1 or frame_num % 12 == 0:
                    det, _ = detector.detect(frame, 0.6)
                    dets = np.copy(det)

                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if dets is None:
                    bboxes = []
                    scores = []
                    classes = []
                    num_objects = 0
                    
                else:   
                    bboxes = np.copy(dets[:,:4])
                    box = np.copy(bboxes)
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
                    feature = encoder(frame, bboxes)
                    features = np.copy(feature)
                
                # if len(features) > 0:
                #     if len(f2) > 0:
                #         check_features = [dot_product(features[i], f2[i]) for i in range(min(len(features), len(f2)))]
                        
                #     f2 = features
                    
                #     if len(check_features) == 0:
                #         print("Start")
                #         for i in range(len(box)):
                #             label = detect.mask_image(frame, model_mask, [box[i]])
                #             if label:
                #                 for j, feature_value in enumerate(feature_mask_values):
                #                     check = dot_product(features[0], feature_value)
                #                     if check > 0.9:
                #                         name.append(idx[j])
                #                         count_check = 0
                #                         break   
                #             else:
                #                 for j, feature_value in enumerate(feature_face_values):
                #                     check = dot_product(features[i], feature_value)
                #                     if check > 0.9 :
                #                         name.append(str(idx[j]))
                #                         count_check = 0
                #                         break
            
                #     else:
                #         for check_feature in check_features:
                #             if check_feature < 0.8:
                #                 check_feature = []
                #                 name = []
                #                 f2 = []
                #                 break
                            
                #     if len(name)==0 and count_check<24:
                #         check_feature = []
                #         f2 = []
                #         # print("Reset")
                #         count_check += 1
                        
                # else:
                #     name = []
                #     check_features = []  
                #     f2 = []
                
                
                # if len(name)!=0:
                #     names = name
                
                # print(name)
                
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
                    
                for track in tracker.tracks:  # update new findings AKA tracks
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue 
                    bbox = track.to_tlbr()
                    class_name = track.get_class()
                    fps = 1.0 / (time.time() - start_time)
                    color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                    color = [i * 255 for i in color]
        
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                    cv2.putText(frame, class_name ,(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                    
                #     if verbose == 2:
                #         print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                        
                # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
               
                print((time.time() - start_time))
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                
                if not count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")
                else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: ")

                # if time.time() - start_fps > 1:
                #     fps_s =  frame_num/(time.time() - start_fps)
                #     print("so khung hinh", fps_s)
                #     frame_num = 0
                #     start_fps = time.time()
                # result = np.asarray(frame)
                
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

class tkSourceSelect(tkinter.Toplevel):

    def __init__(self, parent, other_sources=None):
        """TODO: add docstring"""

        super().__init__(parent)

        self.other_sources = other_sources

        # default values at start
        self.item = None
        self.name = None
        self.source = None

        # GUI
        button = tkinter.Button(self, text="Open file...", command=self.on_select_file)
        button.pack(fill='both', expand=True)

        if self.other_sources:
            tkinter.Label(self, text="Other Sources:").pack(fill='both', expand=True)

            for item in self.other_sources:
                text, source = item
                button = tkinter.Button(self, text=text, command=lambda data=item:self.on_select_other(data))
                button.pack(fill='both', expand=True)

    def on_select_file(self):
        """TODO: add docstring"""

        result = tkinter.filedialog.askopenfilename(
                                        initialdir=".",
                                        title="Select video file",
                                        filetypes=(("AVI files", "*.avi"), ("MP4 files","*.mp4"), ("all files","*.*"))
                                    )

        if result:
            self.item = item
            self.name = name
            self.source = source

            print('[tkSourceSelect] selected:', name, source)

            self.destroy()

    def on_select_other(self, item):
        """TODO: add docstring"""

        name, source = item

        self.item = item
        self.name = name
        self.source = source

        print('[tkSourceSelect] selected:', name, source)

        self.destroy()

class tkCamera(tkinter.Frame):

    def __init__(self, parent, text="", source=0, width=None, height=None, sources=None):
        """TODO: add docstring"""
        tkinter.Frame.__init__(self, parent)
        # super().__init__(parent)

        self.source = source
        self.width  = width
        self.height = height
        self.other_sources = sources

        #self.window.title(window_title)
        self.vid = VideoCapture(video_source=self.source, width=self.width, height=self.height)
        self.vid.start()
        self.label = tkinter.Label(self, text=text)
        self.label.pack()

        self.canvas = tkinter.Canvas(self, width=self.vid.width, height=self.vid.height)
        self.canvas.pack()

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(self, text="Start", command=self.start)
        self.btn_snapshot.pack(anchor='center', side='left')

        self.btn_snapshot = tkinter.Button(self, text="Stop", command=self.stop)
        self.btn_snapshot.pack(anchor='center', side='left')

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(self, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(anchor='center', side='left')

        # Button that lets the user take a snapshot
        self.btn_snapshot = tkinter.Button(self, text="Source", command=self.select_source)
        self.btn_snapshot.pack(anchor='center', side='left')

        # After it is called once, the update method will be automatically called every delay milliseconds
        # calculate delay using `FPS`
        self.delay = int(1000/self.vid.fps)

        print('[tkCamera] source:', self.source)
        print('[tkCamera] fps:', self.vid.fps, 'delay:', self.delay)

        self.image = None

        self.dialog = None

        self.running = True
        self.update_frame()

    def start(self):
        """TODO: add docstring"""

        #if not self.running:
        #    self.running = True
        #    self.update_frame()
        self.vid.start_recording()

    def stop(self):
        """TODO: add docstring"""

        #if self.running:
        #   self.running = False
        self.vid.stop_recording()

    def snapshot(self):
        """TODO: add docstring"""

        self.vid.snapshot()

    def update_frame(self):
        """TODO: add docstring"""

        # widgets in tkinter already have method `update()` so I have to use different name -

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.image = frame
            self.photo = PIL.ImageTk.PhotoImage(image=self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')

        if self.running:
            self.after(self.delay, self.update_frame)

    def select_source(self):
        """TODO: add docstring"""

        # open only one dialog
        if self.dialog:
            print('[tkCamera] dialog already open')
        else:
            self.dialog = tkSourceSelect(self, self.other_sources)

            self.label['text'] = self.dialog.name
            self.source = self.dialog.source

            self.vid = VideoCapture(self.source, self.width, self.height)
            
class App:
    def __init__(self, parent, title, sources):
        """TODO: add docstring"""

        self.parent = parent

        self.parent.title(title)

        self.stream_widgets = []

        width = 400
        height = 300

        columns = 4
        for number, (text, source) in enumerate(sources):
            widget = tkCamera(self.parent, text, source, width, height, sources)
            row = number // columns
            col = number % columns
            widget.grid(row=row, column=col)
            self.stream_widgets.append(widget)

        self.parent.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self, event=None):
        """TODO: add docstring"""

        print("[App] stoping threads")
        for widget in self.stream_widgets:
            widget.vid.running = False

        print("[App] exit")
        self.parent.destroy()
    

if __name__ == "__main__":

    sources = [  # (text, source)
        # local webcams
        # remote videos (or streams)
        (
            "Zakopane, Poland",
            "/home/tima/detec_and_tracking/1.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/5.mp4",
        ),
        
        (
            "Zakopane, Poland",
            "/home/tima/detec_and_tracking/4.mp4",
        ),
        (
            "Zakopane, Poland",
            "/home/tima/detec_and_tracking/1.mp4",
        ),
        (
            "Zakopane, Poland",
            "/home/tima/detec_and_tracking/1.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/5.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/5.mp4",
        ),
        
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/5.mp4",
        ),
          
        # ('Mountains, Poland', 'http://172.16.4.47:4747/video'),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/5.mp4",
        ),
        (
            "Warszawa, Poland",
            "https://vod-progressive.akamaized.net/exp=1690282020~acl=%2Fvimeo-prod-skyfire-std-us%2F01%2F4221%2F14%2F371107124%2F1539961455.mp4~hmac=29ee79702d786b85adc5fee056dba4173b1597326e462a6227c14fc4a9b30928/vimeo-prod-skyfire-std-us/01/4221/14/371107124/1539961455.mp4",
        ),
        (
            "Warszawa, Poland",
            "https://vod-progressive.akamaized.net/exp=1690282065~acl=%2Fvimeo-prod-skyfire-std-us%2F01%2F3440%2F15%2F392201333%2F1661540316.mp4~hmac=fc3d273fdc11d7a2130eba2eebc8862c20cef9cbe1e4f8f47cb1a339c0ff241a/vimeo-prod-skyfire-std-us/01/3440/15/392201333/1661540316.mp4",
        ),
        
        # local files
        # ('2021.01.25 20.37.50.avi', '2021.01.25 20.37.50.avi'),
    ]

    root = tkinter.Tk()
    App(root, "Tkinter and OpenCV", sources)
    root.mainloop()