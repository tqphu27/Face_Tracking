import tkinter

import sys

sys.path.append("/home/tima/detec_and_tracking")

from run.tkCam import *

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
            "/home/tima/detec_and_tracking/1.mp4",
        ),
        
        (
            "Zakopane, Poland",
            "/home/tima/detec_and_tracking/3.mp4",
        ),
        (
            "Zakopane, Poland",
            "/home/tima/detec_and_tracking/6.mp4",
        ),
        
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/10.mp4",
        ),
          
        # ('Mountains, Poland', 'http://172.16.4.47:4747/video'),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/6.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/7.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/8.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/9.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/10.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/1.mp4",
        ),
        (
            "Warszawa, Poland",
            "/home/tima/detec_and_tracking/6.mp4",
        ),
        # (
        #     "Warszawa, Poland",
        #     "/home/tima/detec_and_tracking/9.mp4",
        # ),
        
        # (
        #     "Warszawa, Poland",
        #     "/home/tima/detec_and_tracking/5.mp4",
        # ),
        # (
        #     "Warszawa, Poland",
        #     "https://vod-progressive.akamaized.net/exp=1690282020~acl=%2Fvimeo-prod-skyfire-std-us%2F01%2F4221%2F14%2F371107124%2F1539961455.mp4~hmac=29ee79702d786b85adc5fee056dba4173b1597326e462a6227c14fc4a9b30928/vimeo-prod-skyfire-std-us/01/4221/14/371107124/1539961455.mp4",
        # ),
        # (
        #     "Warszawa, Poland",
        #     "https://vod-progressive.akamaized.net/exp=1690282065~acl=%2Fvimeo-prod-skyfire-std-us%2F01%2F3440%2F15%2F392201333%2F1661540316.mp4~hmac=fc3d273fdc11d7a2130eba2eebc8862c20cef9cbe1e4f8f47cb1a339c0ff241a/vimeo-prod-skyfire-std-us/01/3440/15/392201333/1661540316.mp4",
        # ),
        # local files
        # ('2021.01.25 20.37.50.avi', '2021.01.25 20.37.50.avi'),
    ]

    root = tkinter.Tk()
    App(root, "Tkinter and OpenCV", sources)
    root.mainloop()