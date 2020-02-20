import cv2


def parse():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('input')
    return ap.parse_args()


def main(args):
    from os.path import join, splitext, exists
    from os import mkdir
    output = splitext(args.input)[0]
    if not exists(output):
        mkdir(output)
    cap = cv2.VideoCapture(args.input)
    count = 0
    while 1:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 15 == 0:
            cv2.imwrite(join(output, '%d.jpg' % count), frame)
        count += 1

if __name__ == '__main__':
    main(parse())
