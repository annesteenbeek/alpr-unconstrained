from license_plate_recognition import LicensePlateRecognition

import cv2
import configparser
import os
import csv


def main(config):
    cap = cv2.VideoCapture(config['DEFAULT']['source'])
    estimator = LicensePlateRecognition(config['vehicle'],
                                        config['plate'],
                                        config['ocr'])
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        results = lpr(frame, estimator)
        if len(results) == 0:
            [cap.read() for _ in range(10)]
            continue
        cv2.imshow('lpr_demo', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

def test_dir(config):
    estimator = LicensePlateRecognition(config['vehicle'],
                                        config['plate'],
                                        config['ocr'],
                                        save_img=True)

    sample_folder = "samples"
    result_path = os.path.join(sample_folder, "results")
    result_csv = os.path.join(result_path, 'results.csv')

    with open(result_csv, 'w', newline="") as file:
        writer = csv.writer(file)

        for filename in os.listdir(sample_folder):
        # for filename in ["DJI_0506.JPG"]:
            split = os.path.splitext(filename)
            frame = cv2.imread(os.path.join(sample_folder, filename))
            if frame is None:
                continue

            summary = [filename]

            results = lpr(frame, estimator)
            for i, vehicle in enumerate(results):
                carname = os.path.join(result_path, split[0]+"-vehicle_%d.jpg" % i)
                cv2.imwrite(carname, vehicle.image)
                if vehicle.plate is not None:
                    text = ""
                    if vehicle.plate.text is not None:
                        text = vehicle.plate.text.content
                        summary.append(text)
                        print(text)
                    platename = os.path.join(result_path, split[0]+"-plate_%d_%s.jpg" % (i, text))
                    cv2.imwrite(platename, vehicle.plate.image)
            fname = os.path.join(result_path, split[0]+"_out.jpg")
            cv2.imwrite(fname, frame)

            writer.writerow(summary)


def lpr(frame, estimator):
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
    cp = configparser.ConfigParser()
    cp.read('demo.ini')
    # main(cp)
    test_dir(cp)
