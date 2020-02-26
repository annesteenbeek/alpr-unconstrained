import numpy as np
import tensorflow as tf


def parse():
    from argparse import ArgumentParser
    ap = ArgumentParser()
    ap.add_argument('pb_file_path')
    ap.add_argument('export_dir')
    ap.add_argument('class_path')
    return ap.parse_args()


def main(args):
    with tf.gfile.GFile(args.pb_file_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        image_b = tf.compat.v1.placeholder(tf.string, [])
        image = tf.image.decode_image(image_b)
        image = tf.cast(image, tf.float32)[None]
        outputs_op, = tf.import_graph_def(graph_def,
                                          input_map={'inputs:0': image},
                                          return_elements=['output_boxes:0'],
                                          name="")

        outputs_mtx = tf.reshape(outputs_op, [-1, outputs_op.shape[-1]])
        boxes, obj_scores, cls_scores = tf.split(outputs_mtx, [4, 1, 80], axis=1)

        score_thresh = tf.compat.v1.placeholder(tf.float32, [])
        iou_thresh = tf.compat.v1.placeholder(tf.float32, [])
        max_outputs = tf.compat.v1.placeholder(tf.int32, [])
        keep = tf.image.non_max_suppression(boxes,
                                            obj_scores[:, 0],
                                            max_outputs,
                                            iou_thresh,
                                            score_thresh)
        boxes_fin = tf.gather(boxes, keep)
        obj_scores_fin = tf.gather(obj_scores, keep)
        cls_scores_fin = tf.gather(cls_scores, keep)
        class_name_table = tf.constant(
            list(
                map(str.strip,
                    open(args.class_path).readlines()
                    )
                ), tf.string
            )
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


if __name__ == '__main__':
    main(parse())
