import base64
import cv2
import json
import numpy as np
import requests


def main():
    image_np_org = cv2.imread('samples/ocr/03016_0car_lp.png')
    assert image_np_org is not None
    image_b = cv2.imencode('.jpg', image_np_org)[1]

    response = requests.post(
        'http://localhost:8501/v1/models/ocr:predict',
        json={'inputs':{
                  'image_b':{
                      'b64':base64.b64encode(image_b).decode('utf-8')
                   },
                  'iou_thresh':.2,
                  'max_outputs':8,
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
    for i, box in enumerate(boxes):
        l, t, r, b = map(int, box)
        cv2.rectangle(image_np_org, (l, t), (r, b), (0, 255, 0))
        cv2.putText(image_np_org,
                    '%s' % class_names[i],
                    (l, int(t+b) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
    cv2.imwrite('result.jpg', image_np_org)


if __name__ == '__main__':
    main()
