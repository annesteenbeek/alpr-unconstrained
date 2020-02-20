from cv2 import imread
from os.path import basename, exists
from random import randint


def parse():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('dir_txt')
    ap.add_argument('dir_jpg')
    ap.add_argument('--output', '-o', default='annotation.json')
    return ap.parse_args()

def main(args):
    from glob import glob
    from tqdm import tqdm
    from json import dump
    from os.path import join, splitext
    files = glob(join(args.dir_txt, '*cars.txt'))
    images = {splitext(basename(v))[0]:v for v in glob(join(args.dir_jpg, '*.jpg'))}
    dataset = {}
    categories = []
    for c in ['vehicle', 'plate_single', 'plate_double']:
        c = Category(c)
        categories.append(c)
        dataset.setdefault('categories', []).append(c.__dict__)
    for f in tqdm(files):
        name = splitext(basename(f))[0].replace('_cars', '')
        im = Image(images[name])
        dataset.setdefault('images', []).append(im.__dict__)
        for a in Annotation.read(f, im, categories[0]):
            dataset.setdefault('annotations', []).append(a.__dict__)
    dump(dataset, open(args.output, 'w'), indent=2)


class Entity(object):
    def __init__(self):
        self.id = id(self)

class Image(Entity):
    def __init__(self, path):
        super(type(self), self).__init__()
        self.dataset_id = 0
        self.path = path
        h, w, c = imread(path).shape
        self.width = w
        self.height = h
        self.file_name = basename(path)

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
        # 'segmentation': [[1227.3, 202.5, 1227.3, 490.4, 820.3, 490.4, 820.3, 202.5]],
        # 'area': 116809,
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
                bbox = BBox(map(float, tokens), image)

                lp_path = path.replace('cars.txt', '%scar_lp.txt' % index)
                if exists(lp_path):
                    keypoints = KeyPoint.read(lp_path, bbox)
                else:
                    keypoints = []

                license = ''
                lp_str_path = lp_path.replace('.txt', '_str.txt')
                if exists(lp_str_path):
                    license = open(lp_str_path).readlines()[0].strip()

                a = Annotation(image.id, category.id, bbox,
                               category.color, keypoints,
                               license=license)
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
            s.pop(0)
            for i in range(0, 8, 2):
                x, y = map(float, s[i:i+2])
                keypoints.append(KeyPoint(x, y).abs(bbox))
        return keypoints


class BBox:
    def __init__(self, r_bbox, image):
        width = image.width
        height = image.height
        r_bbox = list(r_bbox)
        self.x = float(r_bbox[0] * width)
        self.y = float(r_bbox[1] * height)
        self.w = float(r_bbox[2] * width)
        self.h = float(r_bbox[3] * height)

    def __iter__(self):
        for a in ['x', 'y', 'w', 'h']:
            yield getattr(self, a)


if __name__ == '__main__':
    main(parse())
