from argparse import ArgumentParser
from ConfigParser import ConfigParser
from json import loads
from tqdm import tqdm

import requests


def parse():
    ap = ArgumentParser()
    ap.add_argument('config')
    return ap.parse_args()

def main(args):
    config = ConfigParser()
    config.read(args.config)
    address = config.get('coco-annotator', 'address')
    port = config.get('coco-annotator', 'port')
    username = config.get('coco-annotator', 'username')
    password = config.get('coco-annotator', 'password')
    url = 'http://{}:{}'.format(address, port)
    login_request = requests.post(url + '/api/user/login',
                                  json={'username': username,
                                        'password': password})
    url_pattern = url + '/api/annotation/'
    url_pattern2 = url + '/api/undo/'
    with tqdm(leave=False, desc='quering annotations') as pbar:
        thing_request = requests.get(url_pattern, cookies=login_request.cookies)
        annotation_json = loads(thing_request.text)
        pbar.total = len(annotation_json)
        url_pattern += '%d'
        pbar.desc = 'deleting annotations'
        for annotation in annotation_json:
            del_req = requests.delete(url_pattern % annotation['id'],
                                      cookies=login_request.cookies)
            del_req2 = requests.delete(url_pattern2,
                                       cookies=login_request.cookies,
                                       params={'id':annotation['id'],
                                               'instance':'annotation'})
            assert del_req2.status_code == 200, del_req2.text
            pbar.update()


if __name__ == '__main__':
    main(parse())
