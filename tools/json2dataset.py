import json, cv2, numpy as np

from tqdm import tqdm
from pdb import set_trace as breakpoint
from os.path import basename, join
from os import walk, linesep


from plate_knowledge import is_valid


def parse():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('json_file')
    ap.add_argument('image_folder')
    ap.add_argument('export_dir')
    return ap.parse_args()


def main(args):
    dataset = json.load(open(args.json_file))
    annotations = dataset['annotations']
    images_json = {i['id']:i['path'] for i in dataset['images']}
    images_disk = {}
    for root, dirs, files in walk(args.image_folder):
        if basename(root).startswith('.'):  # ignore hidden folders
            continue
        for f in files:
            images_disk[f] = join(root, f)

    for a in tqdm(dataset['annotations'], leave=False, desc='exporting files'):
        anno_id = str(a['id'])  # use this to ditinguish examples in the dataset
        img_name = basename(images_json[a['image_id']])
        img_path = images_disk[img_name]

        bbox = a['bbox']
        try:
            keypoints = a['keypoints']
            if len(bbox) != 4:
                continue
            elif len(keypoints) != 12:
                continue
            elif is_valid(a['metadata']['name']):
                save(bbox, keypoints, img_path, join(args.export_dir, anno_id))
        except:
            continue


def save(bbox, keypoints, img_path, export_name, margin=1):
    img_np = cv2.imread(img_path)
    h, w, c = img_np.shape
    l, t, r, b = map(int, bbox)

    points = []
    num_points = len(keypoints) // 3
    for i in range(num_points):
        points.append(keypoints[i * 3])
        points.append(keypoints[i * 3 + 1])
    points = np.array(points, np.float32).reshape([num_points, 2]).T
    xs = points[0]
    ys = points[1]

    l = max(0, min(l, xs.min()) - margin)  # make sure
    t = max(0, min(t, ys.min()) - margin)  # all keypoints
    r = min(w, max(r + l, xs.max()) + margin)  # are in
    b = min(h, max(b + t, ys.max()) + margin)  # the bbox

    try:
        roi_np = img_np[t:b, l:r, :]
        cv2.imwrite(export_name+'.jpg', roi_np)
    except:
        return
    origin = np.array([l, t], np.float32)[:, None]
    scale = np.array([r - l, b - t], np.float32)[:, None]

    points -= origin
    points /= scale
    tokens = [num_points] + points.flatten().tolist()
    label_str = ','.join(map(str, tokens))+',,'
    with open(export_name+'.txt', 'w') as f:
        f.write(label_str + linesep)


if __name__ == '__main__':
    main(parse())
