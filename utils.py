import cv2

BAR_NAME = 0
CURRENT_VALUE = 1
MAX_VALUE = 2

class SliderWindow:
    def __init__(self, name):
        self.name = name
        self.sliders = []
        self.image = None
        self.callback = None

    def add_callback(self, cb):
        self.callback = cb

    def add_image(self, img):
        self.image = img
    
    def add_slider(self, name, default, max_val = 255):
        self.sliders.append((name, default, max_val))

    def show(self):
        assert self.image is not None, "Image must be defined."
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        for s in self.sliders:
            cv2.createTrackbar(
                s[BAR_NAME], 
                self.name, 
                s[CURRENT_VALUE], 
                s[MAX_VALUE], 
                self.on_change)
        self.display_img(self.image)


    def on_change(self, value):
        assert self.callback, "Callback must be defined."
        self.callback(self)

    def get_slider_value(self, key):
        return cv2.getTrackbarPos(key, self.name)

    def close(self):
        cv2.destroyWindow(self.name)
    
    def __enter__(self):        
        return self

    def display_img(self,img):
        cv2.imshow(self.name, img)
    
    def __exit__(self, type, value, traceback):
        self.close()