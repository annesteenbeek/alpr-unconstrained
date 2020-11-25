import base64
import cv2
import json
import numpy as np
import requests

from dataclasses import dataclass
from plate_knowledge import is_valid


@dataclass
class Text:
    content: str
    confidence_lst: list
    is_valid: bool = False


@dataclass
class KeyPoint:
    x: float = 0.
    y: float = 0.


@dataclass
class Plate:
    top_left: KeyPoint = None
    top_right: KeyPoint = None
    bottom_right: KeyPoint = None
    bottom_left: KeyPoint = None
    confidence: float = 0.
    text: Text = None
    image: np.ndarray = None

    def get_keypoints(self):
        keypoints = []
        for n in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            kp = getattr(self, n)
            keypoints.append([kp.x, kp.y])
        return np.array(keypoints, np.int32)


@dataclass
class Vehicle:
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    category: str = ''
    confidence: float = 0.
    plate: Plate = None
    image: np.ndarray = None

    def get_orign(self):
        return np.array([self.left, self.top], np.int32)[None]


class LicensePlateRecognition:
    """
        This class performs the LPR on a given image.
        The vehicle detector may be skipped by choice.
    """
    def __init__(self, vehicle_cfg, plate_cfg, ocr_cfg, save_img=False):
        self.vehicle_detector = VehicleDetection(vehicle_cfg)
        self.plate_detector = PlateDetection(plate_cfg)
        self.ocr = PlateRecognition(ocr_cfg)

        self.save_img = save_img

    def __call__(self, image_np):
        vehicles, images =  self.vehicle_detector(image_np)

        vehicles_w_license = []
        for v, im in zip(vehicles, images):
            if self.save_img: v.image = im
            # if max(im.shape[:2]) < 288: # image too small
                # continue
            plate_and_im = self.plate_detector(im)

            if plate_and_im is not None:
                v.plate = plate_and_im[0]
                image = plate_and_im[1]
                if self.save_img: v.plate.image = image
                text = self.ocr(image)
                if text is not None:
                    v.plate.text = text

            vehicles_w_license.append(v)

        return vehicles_w_license
        


class VehicleDetection:
    def __init__(self, cfg):
        self.score_thresh = float(cfg['score_thresh'])
        self.iou_thresh = float(cfg['iou_thresh'])
        self.max_outputs = int(cfg['max_outputs'])
        self.input_width = int(cfg['input_width'])
        self.input_height = int(cfg['input_height'])
        self.input_size = max(self.input_height, self.input_width)
        self.address = cfg['address'] + '/v1/models/vehicle_detector:predict'
        self.categories = cfg['categories'].split(',')

    def __call__(self, image_np_org):
        h, w = image_np_org.shape[:2]
        if h > w:
            image_np = np.pad(image_np_org, [(0, 0), (0, h - w), (0, 0)],
                              'constant', constant_values=0)
        elif w > h:
            image_np = np.pad(image_np_org, [(0, w - h), (0, 0), (0, 0)],
                              'constant', constant_values=0)
        else:
            image_np = image_np_org.copy()
        image_np = cv2.resize(image_np, (self.input_width, self.input_height))

        image_b = cv2.imencode('.jpg', image_np)[1]
        image_b64 = base64.b64encode(image_b).decode('utf-8')
        inputs = {'image_b': {'b64': image_b64},
                  'iou_thresh': self.iou_thresh,
                  'max_outputs': self.max_outputs,
                  'score_thresh': self.score_thresh}
        response = requests.post(self.address, json={'inputs': inputs})
        assert response.status_code == 200, response.text
        results = json.loads(response.text)['outputs']
        class_names = results['detection_class_names']
        class_confidence = results['detection_class_confidence']
        boxes = np.array(results['detection_boxes'])

        vehicles, image_lst = [], []
        scale = max(h, w) / self.input_size
        for i, name in enumerate(class_names):
            if name in self.categories:
                l, t, r, b = (boxes[i] * scale).astype(np.int32)
                w_, h_ = r - l, b - t
                if w_ < 0 or h_ < 0:
                    continue
                image = np.zeros([h_, w_, 3], np.uint8)  # to fill bbox
                rx1 = ry1 = 0
                rx2 = w_
                ry2 = h_
                if l < 0:
                    rx1 = -l
                    l = 0
                if t < 0:
                    ry1 = -t
                    t = 0
                if r > w:
                    rx2 = w - r  # negative index slicing
                    r = w
                if b > h:
                    ry2 = h - b
                    b = h
                image[ry1:ry2, rx1:rx2, :] = image_np_org[t:b, l:r, :]
                image_lst.append(image)
                vehicles.append(Vehicle(l, t, r, b, name, class_confidence[i]))
        return vehicles, image_lst

    def preprocess(self, x):
        scale = max(h, w) / 416
        Icar = Iorig[max(0, t):min(h, b), max(0, l):min(w, r), :]


class PlateDetection:
    def __init__(self, cfg):
        self.score_thresh = float(cfg['score_thresh'])
        self.iou_thresh = float(cfg['iou_thresh'])
        self.max_outputs = int(cfg['max_outputs'])
        self.address = cfg['address'] + '/v1/models/plate_detector:predict'
        self.out_size = int(cfg['width']), int(cfg['height'])
        w, h = self.out_size
        self.rect_ref_pts = np.matrix([[0, w, w, 0],
                                       [0, 0, h, h]], np.float32).T

    def __call__(self, image):
        image_b = cv2.imencode('.jpg', image)[1]
        image_b64 = base64.b64encode(image_b).decode('utf-8')
        inputs = {'image_b': {'b64': image_b64},
                  'iou_thresh': self.iou_thresh,
                  'max_outputs': self.max_outputs,
                  'score_thresh': self.score_thresh}
        response = requests.post(self.address, json={'inputs': inputs})
        assert response.status_code == 200, response.text
        results = json.loads(response.text)['outputs']
        set_scores = results['scores']
        if len(set_scores):
            keypoints = np.array(results['corners'], np.float32)[0]
            plate = Plate(*[KeyPoint(*pt) for pt in keypoints.T], set_scores[0])
            harmo_mtx = cv2.getPerspectiveTransform(keypoints.T, self.rect_ref_pts)
            # cv2.imshow('v', image)
            # np.save('homograph.npy', harmo_mtx)
            image_rect = cv2.warpPerspective(image, harmo_mtx, self.out_size)
            # cv2.imshow('p', image_rect)
            return plate, image_rect


class PlateRecognition:
    def __init__(self, cfg):
        self.score_thresh = float(cfg['score_thresh'])
        self.iou_thresh = float(cfg['iou_thresh'])
        self.max_outputs = int(cfg['max_outputs'])
        self.address = cfg['address'] + '/v1/models/ocr:predict'
        self.alpha = float(cfg['alpha'])
        self.names = np.array([l.strip() for l in open(cfg['names_file'])])
        prior_csv = cfg.get('prior_csv')
        num_names = len(self.names)
        if prior_csv is None:
            self.prior_mtx = np.ones([num_names, self.max_outputs]) / num_names
        else:
            self.prior_mtx = np.genfromtxt(prior_csv, np.float32, delimiter=',')
            self.prior_mtx /= np.maximum(1, self.prior_mtx.sum(0, keepdims=True))
        self.prior_mtx = self.prior_mtx.T  # [max_outputs, names] format
        self._range = np.arange(self.max_outputs)

    def __call__(self, image):
        image_b = cv2.imencode('.jpg', image)[1]
        image_b64 = base64.b64encode(image_b).decode('utf-8')
        inputs = {'image_b': {'b64': image_b64},
                  'iou_thresh': self.iou_thresh,
                  'max_outputs': self.max_outputs,
                  'score_thresh': self.score_thresh}
        response = requests.post(self.address, json={'inputs': inputs})
        assert response.status_code == 200, response.text
        results = json.loads(response.text)['outputs']
        boxes = np.array(results['detection_boxes'], np.float32)
        class_names = np.array(results['detection_class_names'])
        class_confidence = np.array(results['detection_class_confidence'])
        class_confidence_all = np.array(results['detection_class_confidence_all'])
        # object_scores = results['detection_object_scores']
        if len(boxes):
            # sort by spatial order
            center = boxes.reshape([-1, 2, 2]).mean(1)
            sorting = np.argsort(center[:, 0] + self.alpha * center[:, 1])
            if len(boxes) == self.max_outputs:
                class_confidence_all = class_confidence_all[sorting]
                indices = self.maximize_a_posterior(class_confidence_all)
                text = ''.join(self.names[indices])
                return Text(text,
                            class_confidence_all[self._range, indices], 
                            is_valid(text))
            else:
                return Text(''.join(class_names[sorting]),
                            class_confidence[sorting])

    def maximize_a_posterior(self, likelihood_mtx):
        # use naive bayesian method
        posterior = likelihood_mtx * self.prior_mtx
        assign = np.argmax(posterior, 1)
        return assign
