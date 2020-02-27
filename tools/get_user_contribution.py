from argparse import ArgumentParser
from ConfigParser import ConfigParser
from pdb import set_trace as breakpoint
from tqdm import tqdm
from plate_knowledge import is_valid

import requests, json, pickle


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

    contribution_dict = {}  # user, action -> number

    thing_request = requests.get(url_pattern, cookies=login_request.cookies)
    annotation_json = json.loads(thing_request.text)
    pbar.total = len(annotation_json)
    pbar.desc = 'deleting annotations'
    for annotation in annotation_json:
        for event in annotation.get('events', []):
            tools = event['tools_used']
            user = event['user']
            date = event['created_at']['$date']
            for t in tools:
                key = (t, user)
                if key in contribution_dict:
                    contribution_dict[key] += 1
                else:
                    contribution_dict[key] = 1
    pickle.dump(contribution_dict, open('contribution_dict.pkl', 'wb'))
    row_names = set()
    clm_names = set()
    for k in contribution_dict.keys():
        c, r = k
        row_names.add(r)
        clm_names.add(c)
    clm_names = list(clm_names)
    print('\t'+'\t'.join(clm_names))
    for r in row_names:
        line = r
        for c in clm_names:
            v = contribution_dict.get((c,r), 0)
            line += '\t{}'.format(v)
        print(line)


if __name__ == '__main__':
    main(parse())
