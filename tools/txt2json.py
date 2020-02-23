from argparse import ArgumentParser
from ConfigParser import ConfigParser
from glob import glob
from io import BytesIO
from json import loads, dumps
from os.path import exists, join
from pdb import set_trace as breakpoint
from random import randint
from tqdm import tqdm

import requests


def parse():
    ap = ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('txt_dir')
    ap.add_argument('dataset_id', type=int)
    return ap.parse_args()


def main(args):
    url, auth_cookies = login(args.config)

    dataset = {'images':[], 'categories':[], 'annotations':[]}
    dataset_id = args.dataset_id

    categories = []
    url_cate = url + '/api/category/'
    for c in ['vehicle_single_plate', 'vehicle_double_plate']:
        c = Category(c)
        categories.append(c)
        c.metadata.update({'name':''})
        dataset['categories'].append(c.__dict__)

    page = 1
    pages = None
    url_pattern = url + '/api/image/?page=%d&per_page=50'
    url_new = url + '/api/annotation/'
    with tqdm(unit='pages', leave=False, desc='annotating images') as pbar:
        while 1:
            thing_request = requests.get(url_pattern % page, cookies=auth_cookies)
            assert thing_request.status_code == 200, thing_request.text
            image_json = loads(thing_request.text)
            if pages is None:
                pages = image_json['pages']
                pbar.total = pages
            page += 1
            pbar.update()
            if page > pages:
                break

            for i in image_json['images']:
                if i['dataset_id'] != dataset_id:
                    Warning('image from another dataset')
                    continue
                image_name = i['file_name']
                annot_name = image_name.replace('.jpg', '_cars.txt')
                annot_path = join(args.txt_dir, annot_name)
                if not exists(annot_path):
                    continue
                im = Image(**i)
                dataset['images'].append(im.__dict__)
                for a in Annotation.read(annot_path, im, categories[0]):
                    dataset['annotations'].append(a.__dict__)
            upload(url, auth_cookies, dataset_id, dataset)    # per page


def login(config_path):
    config = ConfigParser()
    config.read(config_path)
    address = config.get('coco-annotator', 'address')
    port = config.get('coco-annotator', 'port')
    username = config.get('coco-annotator', 'username')
    password = config.get('coco-annotator', 'password')
    url = 'http://{}:{}'.format(address, port)
    login_request = requests.post(url + '/api/user/login',
                                  json={'username': username,
                                        'password': password})
    assert login_request.status_code == 200, login_request.text
    auth_cookies = login_request.cookies
    return url, auth_cookies


def upload(url, auth, dataset_id, dataset):
    coco_req = requests.post(url + '/api/dataset/%d/coco' % dataset_id,
                             cookies=auth,
                             files={'coco':BytesIO(dumps(dataset))})
    assert coco_req.status_code == 200, coco_req.text
    dataset['images'] = []
    dataset['annotations'] = []


class Entity(object):
    def __init__(self):
        self.id = id(self)


class Image(Entity):
    def __init__(self, **kwargs):
        super(type(self), self).__init__()
        self.id = kwargs['id']
        self.dataset_id = kwargs['dataset_id']
        self.path = kwargs['path']
        self.width = kwargs['width']
        self.height = kwargs['height']
        self.file_name = kwargs['file_name']


class Category(Entity):
    def __init__(self, name, supercategory='', **metadata):
        super(type(self), self).__init__()
        self.name = name
        self.supercategory = supercategory
        def random_byte():
            return '%02X' % randint(0, 255)
        self.color = '#' + ''.join([random_byte()for _ in '123'])
        self.metadata = metadata


class Annotation(Entity):
    def __init__(self, image_id, category_id, bbox, color,
                 keypoints=[], iscrowd=False, isbbox=True, **metadata):
        super(type(self), self).__init__()
        self.image_id = image_id
        self.category_id = category_id
        self.bbox = list(bbox)
        self.iscrowd = iscrowd
        self.isbbox = isbbox
        self.color = color
        xs = self.bbox[0], self.bbox[0] + self.bbox[2]
        ys = self.bbox[1], self.bbox[1] + self.bbox[3]
        def points(i):
            '''
            every 2 numbers represent a point
            points are in the order as the following
                3.....0
                :     :
                :     :
                2.....1
            '''
            z = i // 2
            if i % 2:
                if z // 2:
                    return ys[1 - z % 2]
                else:
                    return ys[z % 2]
            else:
                return xs[1 - z // 2]
        self.segmentation = [list(map(points, range(8)))]
        self.area = self.bbox[2] * self.bbox[3]
        self.keypoints = []
        for k in keypoints:
            assert isinstance(k, KeyPoint)
            self.keypoints.extend(list(k))
        self.num_keypoints = len(keypoints)
        self.metadata = metadata

    @staticmethod
    def read(path, image, category):
        annotations = []
        with open(path) as f:
            for l in f:
                tokens = l.strip().split()
                index = tokens.pop(0)
                r_bbox = list(map(float, tokens))
                # yolo offset from center
                r_bbox[0] -= r_bbox[2] / 2
                r_bbox[1] -= r_bbox[3] / 2
                bbox = BBox(r_bbox, image)

                lp_path = path.replace('cars.txt', '%scar_lp.txt' % index)
                if exists(lp_path):
                    keypoints = KeyPoint.read(lp_path, bbox)
                else:
                    keypoints = []

                liscense = ''
                lp_str_path = lp_path.replace('.txt', '_str.txt')
                if exists(lp_str_path):
                    liscense = open(lp_str_path).readlines()[0].strip()

                a = Annotation(image.id, category.id, bbox,
                               category.color, keypoints,
                               name=liscense or 'null')
                annotations.append(a)
        return annotations


class KeyPoint:
    def __init__(self, x, y, visible=True, labeled=True):
        self.x = x
        self.y = y
        if not labeled:
            self.code = 0
        elif visible:
            self.code = 2
        else:
            self.code = 1

    def abs(self, bbox):
        self.x = bbox.x + self.x * bbox.w
        self.y = bbox.y + self.y * bbox.h
        return self

    def __iter__(self):
        for a in ['x', 'y', 'code']:
            yield getattr(self, a)

    @staticmethod
    def read(path, bbox):
        keypoints = []
        with open(path) as f:
            s = f.readlines()[0].strip().split(',')
            points = int(s.pop(0))
            for i in range(points):
                x = float(s[i])
                y = float(s[i + points])
                keypoints.append(KeyPoint(x, y).abs(bbox))
        return keypoints


class BBox:
    def __init__(self, r_bbox, image):
        width = image.width
        height = image.height
        self.x = float(r_bbox[0] * width)
        self.y = float(r_bbox[1] * height)
        self.w = float(r_bbox[2] * width)
        self.h = float(r_bbox[3] * height)

    def __iter__(self):
        for a in ['x', 'y', 'w', 'h']:
            yield getattr(self, a)


if __name__ == '__main__':
    main(parse())
