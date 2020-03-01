import sys
import cv2
import numpy as np
import traceback

import requests
import json
import base64

from pdb import set_trace as breakpoint

# import darknet.python.darknet as dn

from src.label 				import Label, lwrite
from os.path 				import splitext, basename, isdir
from os 					import makedirs
from src.utils 				import crop_region, image_files_from_folder
# from darknet.python.darknet import detect


def detect(image_np_org, score_thresh, nms_iou=.45):
	h, w = image_np_org.shape[:2]
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
				'b64':base64.b64encode(image_b).decode('utf-8')},
			'iou_thresh':nms_iou,
			'max_outputs':32,
			'score_thresh':score_thresh
			}
		}
	)
	assert response.status_code == 200, response.text
	results = json.loads(response.text)['outputs']
	R = [
		results['detection_class_names'],
		results['detection_class_confidence'],
		np.array(results['detection_boxes']) / 416. * max(h, w)
		]
	return list(zip(*R))


if __name__ == '__main__':

	try:

		input_dir  = sys.argv[1]
		output_dir = sys.argv[2]
		categories = ['car','bus', 'truck']

		vehicle_threshold = .5

		# vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
		# vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
		# vehicle_dataset = 'data/vehicle-detector/voc.data'

		imgs_paths = image_files_from_folder(input_dir)
		imgs_paths.sort()

		if not isdir(output_dir):
			makedirs(output_dir)

		print 'Searching for vehicles using YOLO...'

		for i,img_path in enumerate(imgs_paths):

			print '\tScanning %s' % img_path

			bname = basename(splitext(img_path)[0])
			Iorig = cv2.imread(img_path)
			WH = Iorig.shape[1::-1]
			R = detect(Iorig , vehicle_threshold, .3)
			R = [r for r in R ]

			print '\t\t%d cars found' % len(R)

			Lcars = []
			for i,r in enumerate(R):
				name = r[0]
				if name in categories:
					prob = r[1]
					# l,t,r,b = map(int, r[2])
					ltrb = np.array(r[2]) / np.concatenate([WH,WH])
					tl, br = np.split(ltrb, [2])
					# tl = np.array([cx - w/2., cy - h/2.])
					# br = np.array([cx + w/2., cy + h/2.])
					label = Label(0, tl, br)
					Icar = crop_region(Iorig, label)
					Lcars.append(label)
					# Icar = Iorig[max(0, t):min(h, b), max(0, l):min(w, r), :]
					cv2.imwrite('%s/%s_%dcar.png' % (output_dir, bname,i), Icar)
			# cv2.rectangle(Iorig, (l,t), (r,b), (0,255,0))
			# cv2.imshow('demo', Iorig)
			# if cv2.waitKey(0) == 27:
			# 	break
			if len(Lcars):
				lwrite('%s/%s_cars.txt' % (output_dir, bname), Lcars)
	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
