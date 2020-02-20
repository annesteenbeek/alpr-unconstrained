import json


def parse():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('json_file')
    return ap.parse_args()

def main(args):
    dataset = json.load(open(args.json_file))
    images = {i['id']:i for i in dataset['images']}
    useful = set()
    for a in dataset['annotations']:
        useful.add(images[a['image_id']]['path'])
    for u in useful:
        print(u)

if __name__ == '__main__':
    main(parse())
