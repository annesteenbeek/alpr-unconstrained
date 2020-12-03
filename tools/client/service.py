import base64
from license_plate_recognition import LicensePlateRecognition
import cv2
import configparser
import os
import logging
from flask import Flask, request
import numpy as np
from dataclasses import asdict
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)

PORT = os.getenv('PORT', 6969)

app = Flask(__name__)
estimator = None

@app.route('/detect_licenses' , methods=['POST'])
def mask_image():
    logger.info("Received post from: " + request.host_url)
    file = request.files['image'].read() ## byte file
    form = request.form

    return_image = form['return_image'] == "true"

    npimg = np.fromstring(file, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)


    results = lpr(img)
    licenses = []
    for i, vehicle in enumerate(results):
        if vehicle.plate is not None and vehicle.plate.text is not None:
            text = vehicle.plate.text.content
    
    def convert(o):
        if type(o).__module__ == np.__name__:
            if type(o) is np.ndarray:
                return o.tolist()
            return o.item()


    img_str = None
    if return_image:
        _, buffer = cv2.imencode('.jpg', img)
        b64_img = base64.b64encode(buffer)
        img_str = b64_img.decode('utf-8')

    json_results = json.dumps({"results" : [asdict(v) for v in results],
                               "image": img_str}, default=convert)

    return json_results 

def lpr(frame):
    '''perform lpr and draw results on the given image'''
    results = estimator(frame)
    for vehicle in results:
        cv2.rectangle(frame,
                      (vehicle.left, vehicle.top),
                      (vehicle.right, vehicle.bottom),
                      (0, 255, 0), 2)
        if vehicle.plate is not None:
            kpt = vehicle.plate.get_keypoints()
            orign = vehicle.get_orign()
            kpt += orign
            cv2.polylines(frame,
                          [kpt[:, None]],
                          True, (0, 0, 255), 1)
            if vehicle.plate.text is not None:
                l, t = kpt.min(0)
                font_scale = 1
                thickness = 2
                (w, h) = cv2.getTextSize(vehicle.plate.text.content,
                                         cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=font_scale,
                                         thickness=thickness)[0]
                if vehicle.plate.text.is_valid:
                    color = (255, 0, 255)
                else:
                    color = (255, 255, 0)
                cv2.rectangle(frame, (l, t - h), (l + w, t),
                              color, cv2.FILLED)
                cv2.putText(frame, vehicle.plate.text.content,
                            (l, t), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 255), thickness)
    return results


if __name__ == '__main__':
    logger.info("Starting service")
    config = configparser.ConfigParser()
    config.read('demo.ini')

    estimator = LicensePlateRecognition(config['vehicle'],
                                        config['plate'],
                                        config['ocr'])

    app.run(host='0.0.0.0', port=PORT)

