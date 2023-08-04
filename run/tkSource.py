import tkinter
import tkinter.filedialog

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