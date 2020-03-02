from license_plate_recognition import LicensePlateRecognition

import cv2
import configparser


def main(config):
    cap = cv2.VideoCapture(config['DEFAULT']['source'])
    estimator = LicensePlateRecognition(config['vehicle'],
                                        config['plate'],
                                        config['ocr'])
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        lpr(frame, estimator)
        cv2.imshow('lpr_demo', frame)
        if 27 == cv2.waitKey(0):
            break


def lpr(frame, estimator):
    '''perform lpr and draw results on the given image'''
    results = estimator(frame)
    # if len(results): print(results)
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
                cv2.rectangle(frame, (l, t - h), (l + w, t),
                              (255, 0, 255), cv2.FILLED)
                cv2.putText(frame, vehicle.plate.text.content,
                            (l, t), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 255, 255), thickness)


if __name__ == '__main__':
    cp = configparser.ConfigParser()
    cp.read('demo.ini')
    main(cp)
