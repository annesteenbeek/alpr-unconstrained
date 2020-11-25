import base64
import cv2
import json
import numpy as np
import requests


def main():
    # image_np_org = cv2.imread('samples/train-detector/00024.jpg')
    image_np_org = cv2.imread('samples/train-detector/vehicle2.jpg')
    # image_np_org = cv2.imread('samples/train-detector/vehicle.png')
    assert image_np_org is not None
    # preprocessing
    image_shape = image_np_org.shape[:2]
    max_side = max(image_shape)
    min_side = min(image_shape)
    if max_side > 608:
        factor = 608. / max_side
        image_np = cv2.resize(image_np_org, (0,0), fx=factor, fy=factor)
    elif min_side > 288:
        image_np = image_np_org
    # elif max_side < 288:
        # print('image too small')
        # return
    image_np = image_np_org.copy()
    # making a request
    image_b = cv2.imencode('.jpg', image_np)[1]
    # image = open('samples/train-detector/00011.jpg', 'rb').read()
    # image_b = np.array(bytearray(image), np.uint8)
    response = requests.post(
        'http://localhost:8501/v1/models/plate_detector:predict',
        json={'inputs':{
                  'image_b':{
                      'b64':base64.b64encode(image_b).decode('utf-8')
                   },
                  'iou_thresh':.1,
                  'max_outputs':4,
                  'score_thresh':.001
                  }
            }
    )
    assert response.status_code == 200, response.text
    # postprocessing
    results = json.loads(response.text)['outputs']
    print(results)
    pts = np.array(results['corners'])
    scrs = results['scores']
    # visualization
    for s, p in zip(scrs, pts.astype(np.int32)):
        cv2.polylines(image_np, [p.T[:, None]], True, (0, 255, 0), 2)
        l, t = p[:, 0]
        cv2.putText(image_np, '%.2f' % s, (l, t),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imwrite('result.jpg', image_np)


if __name__ == '__main__':
    main()
