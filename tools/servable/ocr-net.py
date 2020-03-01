import numpy as np
import tensorflow as tf

from pdb import set_trace as breakpoint


def parse():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('pb_file_path')
    ap.add_argument('export_dir')
    ap.add_argument('class_path')
    ap.add_argument('--anchors', default='3.638,5.409,3.281,4.764')
    ap.add_argument('--net_stride', default=8)
    return ap.parse_args()


def main(args):
    with tf.gfile.GFile(args.pb_file_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        image_b = tf.compat.v1.placeholder(tf.string, [])
        image = tf.image.decode_image(image_b)
        # image = tf.cast(image, tf.float32)
        image = tf.image.convert_image_dtype(image, tf.float32)
        node_names = [n.name for n in graph_def.node]
        outputs_op, = tf.import_graph_def(graph_def,
                                          input_map={'input:0': image[None]},
                                          return_elements=['output:0'],
                                          name="")
        anchors = np.array([float(s) for s in args.anchors.split(',')])
        class_name_table = tf.constant(
            list(
                map(str.strip,
                    open(args.class_path).readlines()
                    )
                ), tf.string
            )
        outputs_lst = detection_layer(outputs_op[0],
                                      class_name_table.shape[0].value,
                                      anchors.reshape([-1, 2]),
                                      [args.net_stride] * 2)
        box_centers, box_sizes, obj_scores, cls_scores = outputs_lst
        box_offsets = box_sizes / 2
        boxes = tf.concat([box_centers - box_offsets, box_centers + box_offsets], axis=-1)

        score_thresh = tf.compat.v1.placeholder(tf.float32, [])
        iou_thresh = tf.compat.v1.placeholder(tf.float32, [])
        max_outputs = tf.compat.v1.placeholder(tf.int32, [])
        obj_scores = obj_scores[:, 0]
        keep = tf.image.non_max_suppression(boxes,
                                            obj_scores,
                                            max_outputs,
                                            iou_thresh,
                                            score_thresh)
        boxes_fin = tf.gather(boxes, keep)
        obj_scores_fin = tf.gather(obj_scores, keep)
        cls_scores_fin = tf.gather(cls_scores, keep)

        indices = tf.argmax(cls_scores_fin, 1)
        class_confidence = tf.reduce_max(cls_scores_fin, 1)
        class_names = tf.gather(class_name_table, indices)
        with tf.Session(graph=graph) as sess:
            tf.saved_model.simple_save(sess,
                                       args.export_dir,
                                       inputs={'image_b':image_b,
                                               'max_outputs': max_outputs,
                                               'iou_thresh':iou_thresh,
                                               'score_thresh':score_thresh},
                                       outputs={'detection_class_names':class_names,
                                                'detection_object_scores':obj_scores_fin,
                                                'detection_boxes':boxes_fin,
                                                'detection_class_confidence':class_confidence},
                                       legacy_init_op=tf.tables_initializer()
                                       )


def detection_layer(predictions, num_classes, anchors, stride):
    num_anchors = len(anchors)
    grid_size = tf.shape(predictions)[:2]
    dim = grid_size[0] * grid_size[1]
    bbox_attrs = 5 + num_classes
    anchors = [(a[0], a[1]) for a in anchors]

    predictions = tf.reshape(predictions, [num_anchors * dim, bbox_attrs])
    box_centers, box_sizes, confidence, classes = tf.split(
        predictions, [2, 2, 1, num_classes], axis=-1)

    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)

    grid_x = tf.range(grid_size[1])
    grid_y = tf.range(grid_size[0])
    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.tile(x_y_offset, [1, num_anchors])
    x_y_offset = tf.reshape(x_y_offset, [-1, 2])
    x_y_offset = tf.cast(x_y_offset, tf.float32)

    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors
    box_sizes = box_sizes * stride
    classes = tf.nn.sigmoid(classes)
    return box_centers, box_sizes, confidence, classes


if __name__ == '__main__':
    main(parse())
