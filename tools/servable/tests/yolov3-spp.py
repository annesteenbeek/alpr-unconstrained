import base64
import cv2
import json
import numpy as np
import requests


def main():
    image_np_org = cv2.imread('samples/train-detector/00024.jpg')
    assert image_np_org is not None

    h, w = image_np_org.shape[:2]
    scale = max(h, w) / 416.
    if h > w:
        p = h - w
        image_np_pad = np.pad(image_np_org, [(0, 0), (0, p), (0, 0)], 'constant', constant_values=0)
    elif h < w:
        p = w - h
        image_np_pad = np.pad(image_np_org, [(0, p), (0, 0), (0, 0)], 'constant', constant_values=0)
    else:
        image_np_pad = image_np_org
    image_np = cv2.resize(image_np_pad, (416,)*2)
    image_b = cv2.imencode('.jpg', image_np)[1]

    response = requests.post(
        'http://localhost:8501/v1/models/vehicle_detector:predict',
        json={'inputs':{
                  'image_b':{
                      'b64':base64.b64encode(image_b).decode('utf-8')
                   },
                  'iou_thresh':.5,
                  'max_outputs':32,
                  'score_thresh':.5
                  }
            }
    )
    assert response.status_code == 200, response.text

    results = json.loads(response.text)['outputs']
    boxes = np.array(results['detection_boxes'])
    class_names = results['detection_class_names']
    class_confidence = results['detection_class_confidence']
    object_scores = results['detection_object_scores']

    for i, box in enumerate(boxes * scale):
        l,t,r,b = map(int, box)
        cv2.rectangle(image_np_org, (l,t), (r,b), (0,255,0))
        cv2.putText(image_np_org,
                    '%s:%f' % (class_names[i], class_confidence[i]),
                    (int(l+r)//2, int(t+b)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imwrite('result.jpg', image_np_org)


if __name__ == '__main__':
    main()
