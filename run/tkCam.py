import tkinter
import PIL.Image
import PIL.ImageTk
import sys

sys.path.append("/home/tima/detec_and_tracking")
from run.tkSource import *
from run.VideoCapture import *


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
        