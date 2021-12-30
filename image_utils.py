import cv2
import numpy as np

def resize_keep_aspect_ratio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter), 1/r

def show_image(window_name, img, width=300):
    scaled_im, scale = resize_keep_aspect_ratio(img, width)
    cv2.imshow(window_name, scaled_im)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

# returns roi patch
def show_img_select_rois(window_name, img, same_size=False, single=False, width=300):
    rs = show_img_select_roi_coords(window_name, img, single=single)
    rs = [rs] if single else rs

    chunks = []

    for r in rs:
        right, top, left, bottom = r[0], r[1], r[0]+r[2], r[1]+r[3]
        chunk = img[top:bottom, right:left]
        top = img.shape[0] - chunk.shape[0]
        left =  img.shape[1] - chunk.shape[1]
        if same_size:
            chunk = cv2.copyMakeBorder(chunk,top,0,left,0,cv2.BORDER_CONSTANT)
        chunks.append(chunk)
    
    return chunks[0] if single else chunks

def show_img_select_roi_coords(window_name, img, single=False, width=300):
    scaled_im, scale = resize_keep_aspect_ratio(img, width)
    
    if single:
        r = cv2.selectROI(window_name, scaled_im, False) # fromCenter = False4
        rs = [r]
    else:
        rs = cv2.selectROIs(window_name, scaled_im, False) # fromCenter = False

    scaled_rs = []
    for r in rs:
        scaled_r = int(scale*r[0]), int(scale*r[1]), int(scale*r[2]), int(scale*r[3])
        scaled_rs.append(scaled_r)

    cv2.destroyWindow(window_name)

    return scaled_rs[0] if single else scaled_rs

def show_image_select_points(window_name, img, width=300, colors=[[255,0,0]], uv=False):
    scaled_im, scale = resize_keep_aspect_ratio(img, width)
    print(colors[0])

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            im_shape = params['img'].shape
            if params['uv']:
                coord = (y/im_shape[0], x/im_shape[1])
            else:
                coord = (int(y*params['scale']), int(x*params['scale']))
            params['points'].append(coord)
            color_index = params['count'] % len(params['colors'])
            color = params['colors'][color_index]
            cv2.circle(params['img'], (x,y), 5, color, -1)
            cv2.imshow(window_name, params['img'])
            params['count']+=1
            print(params['colors'][(color_index+1) % len(params['colors'])])

    params = {
        'img':scaled_im,
        'scale':scale,
        'count':0,
        'points':[],
        'colors': colors,
        'uv': uv
    }

    cv2.imshow(window_name, scaled_im)
    cv2.setMouseCallback(window_name, click_event, params)

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    return params['points']

def draw_text_on_image(img, string, location=(0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, string, location, font,
        1, (255, 0, 0), 2)

def white_balance(img, method='mean', percentile_value=.99, patch=None):
    if method == 'greyworld':
        img_mean = (img * (img.mean() / img.mean(axis=(0, 1))))
    elif method == 'patch':
        assert patch is not None, 'Patch argument cannot be None.'
        img_mean = (img*1.0 / patch.max(axis=(0, 1)))
        img_mean = 255*img_mean.clip(0,1)
    else:
        denom = 1
        if method == 'mean':
            denom = img.mean(axis=(0,1))
        elif method == 'max':
            denom = img.max(axis=(0,1))
        elif method == 'percentile':
            denom = np.percentile(img, percentile_value, axis=(0, 1))
        img_mean = (img*1.0 / denom)
    
    #img_mean.clip(0, 1)
    return img_mean.astype('uint8')

def open_and_white_balance(img_name, hsv=False):
    img = cv2.imread(img_name)
    white_patch = show_img_select_rois('Select White', img, single=True)
    balanced_img = white_balance(img, method='patch', patch=white_patch)
    return cv2.cvtColor(balanced_img, cv2.COLOR_BGR2HSV) if hsv else balanced_img

def square_filter(img, position, radius):
    mask = np.zeros_like(img)
    mask = cv2.rectangle(mask, [position[0]-radius, position[1]-radius], [position[0]+radius, position[1]+radius], (255,255,255), -1)
    return cv2.bitwise_and(img, mask)

def clamp(val, mmin, mmax):
    return mmin if val < mmin else (mmax if val > mmax else val)

def extract_square_patch(img, position, radius):
    vert = slice(
        clamp(position[0]-radius, 0, img.shape[0]), 
        clamp(position[0]+radius, 0, img.shape[0]))
    horis = slice(
        clamp(position[1]-radius, 0, img.shape[1]), 
        clamp(position[1]+radius, 0, img.shape[1]))
    return img[vert, horis]

def extract_img_min_and_max(img):
    lower = np.amin(img, axis=(0,1))
    upper = np.amax(img, axis=(0,1))
    return lower, upper 

def create_forgiving_bounding_box(lower_hsv, upper_hsv, hue_forgiveness=5):
    lower = np.array([
        lower_hsv[0]-hue_forgiveness,
        100,
        50,
    ])
    upper =np.array([
        upper_hsv[0]+hue_forgiveness,
        255,
        255,
    ])
    return lower, upper

def hsv_color_filter(hsv_img, hue_forgiveness=5):
    lower, upper = extract_img_min_and_max(hsv_img)
    return create_forgiving_bounding_box(lower, upper, hue_forgiveness)

def color_filter_mask(hsv_img, lower_color, upper_color, inv=False):
    #create a mask for green colour using inRange function
    mask = cv2.inRange(hsv_img, lower_color, upper_color)
    #perform bitwise and on the original image arrays using the mask
    #res = cv2.bitwise_and(im2, im2, mask=mask)
    return cv2.bitwise_not(mask) if inv else mask

def color_hsv_epsilon(color, eh, es, ev):
    lower = np.array([color[0] - eh, color[1] - es, color[2] - ev])
    upper = np.array([color[0] + eh, color[1] + es, color[2] + ev])

    return lower, upper

def detect_circular_blobs(img):
    params = cv2.SimpleBlobDetector_Params()

    # squareness = 1 rectangular-ness = 0
    params.filterByInertia = False

    # shape/convex hull of shape
    params.filterByConvexity = False

    # number of pixels
    params.filterByArea = True
    params.minArea = 150

    # 4*pi*area/perimeter^2
    params.filterByCircularity = True
    params.minCircularity = 0.75

    detector = cv2.SimpleBlobDetector_create(params)

    return detector.detect(img)

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.legacy.TrackerKCF_create,
    "boosting": cv2.legacy.TrackerBoosting_create,
    "mil": cv2.legacy.TrackerMIL_create,
    "tld": cv2.legacy.TrackerTLD_create,
    "medianflow": cv2.legacy.TrackerMedianFlow_create,
    "mosse": cv2.legacy.TrackerMOSSE_create
}

class ObjectTracker:
    def __init__(self, tracker_type='csrt', draw_debug=False, colors=[(0, 255, 0)]):
        self.trackers = cv2.legacy.MultiTracker_create()
        self.draw_debug = draw_debug
        self.tracker_type = tracker_type
        self.colors = colors

    @classmethod
    def from_boxes(cls, img, boxes, **kwargs):
        track = cls(**kwargs)
        for box in boxes:
            track.add_tracker_from_img_box(img, box)
        return track

    def update(self, img):
        success, boxes = self.trackers.update(img)
        if success:
            if self.draw_debug:
                for i, box in enumerate(boxes):
                    (x, y, w, h) = [int(v) for v in box]
                    color = self.colors[i % len(self.colors)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        return boxes if success else []

    def add_new_tracker(self, img, window_name="Select Object To Track", width=600):
        box = show_img_select_roi_coords(window_name, img, single=True, width=width)
        self.add_tracker_from_img_box(img, box)
        return box
    
    def add_tracker_from_img_box(self, img, box):
        tracker = OPENCV_OBJECT_TRACKERS[self.tracker_type]()
        self.trackers.add(tracker, img, box)
        # tracker.save("default_csrt.xml")
        # fs = cv2.FileStorage("default_csrt.xml", cv2.FILE_STORAGE_READ)
        # fn = fs.getFirstTopLevelNode()
        # tracker.read(fn)


    def save(self, location):
        for tracker in self.trackers:
            tracker.write(location)

def center_of_box(box):
    x, y, w, h = box
    return [x + w/2, y + h/2]

def draw_points_on_image(im, uv_points, radius=10, colors = [[255, 255, 0]]):
    img = im
    count = 0
    for point in uv_points:
        if point[0] < 0 or 1 < point[0] or point[1] < 0 or 1 < point[1]:
            continue

        y = int(point[0] * img.shape[0])
        x = int(point[1] * img.shape[1])

        img = cv2.circle(img, (x,y), radius=radius, color=colors[count % len(colors)], thickness=-1)
        count += 1
    return img