import tensorflow as tf
import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from pdb import set_trace as breakpoint

import sys

path = sys.argv[1]
export_dir = sys.argv[2]
keras.backend.set_learning_phase(0)

with open('%s.json' % path) as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

with keras.backend.get_session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    model.load_weights('%s.h5' % path)
    output_node_name = model.output.name.split(':')[0]
    graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        sess.graph.as_graph_def(),
        [output_node_name]
    )

with tf.Graph().as_default() as graph:
    image_b = tf.compat.v1.placeholder(tf.string, [])
    image_u = tf.image.decode_image(image_b)
    image_u = image_u[...,::-1]     # this keras model takes BGR images
    image_f = tf.image.convert_image_dtype(image_u, tf.float32)[None]
    outputs_op, = tf.import_graph_def(graph_def,
                                      input_map={model.input.name: image_f},
                                      return_elements=[model.output.name],
                                      name="")
    output = outputs_op[0]            #-1, -1, 8, downsized 2 ^ 4
    probs, _, affine = tf.split(output, [1, 1, 6], axis=2)
    probs = probs[..., 0]
    score_thresh = tf.compat.v1.placeholder(tf.float32, [])  # arg
    keep = tf.where(probs > score_thresh)
    affine = tf.gather_nd(affine, keep) # [N, 6]
    scores = tf.gather_nd(probs, keep)  # [N]
    # # restrict affine transform from mirroring and flipping
    affine = tf.split(affine, [1] * 6, axis=1)
    affine[0] = tf.maximum(affine[0], 0.)
    affine[4] = tf.maximum(affine[4], 0.)
    affine_1 = tf.concat(affine[:3], axis=1)
    affine_2 = tf.concat(affine[3:], axis=1)
    affine = tf.stack([affine_1, affine_2], axis=1)
    # # define regression base
    vxx = vyy = 0.5 #alpha
    base = lambda vx,vy: np.matrix([[-vx,-vy,1.],
                                 [vx,-vy,1.],
                                 [vx,vy,1.],
                                 [-vx,vy,1.]], np.float32).T
    base = base(vxx, vyy)[None]
    imaginary_box = tf.constant(base)

    pts = tf.matmul(affine, imaginary_box)
    net_stride = 2. ** 4
    pts_MN_center_mn = pts * (208. + 40.) / 2.
    coords = tf.cast(keep[:,::-1,None], tf.float32)  # x, y coords from indices
    coords += .5    # correction term
    pts_MN = pts_MN_center_mn + coords * net_stride
    pts_prop = pts_MN # output, in pixels
    # # nms using up-right minimal surrounding rectangle representation
    boxes = tf.gather(pts_prop, [0,2], axis=-1)
    boxes = tf.reshape(boxes, [-1, 4])
    iou_thresh = tf.compat.v1.placeholder(tf.float32, [])   # arg
    max_outputs = tf.compat.v1.placeholder(tf.int32, [])     # arg
    keep = tf.image.non_max_suppression(boxes, scores, max_outputs, iou_thresh)
    pts_fin = tf.gather(pts_prop, keep) # output
    scores_fin = tf.gather(scores, keep) # output

    with tf.compat.v1.Session() as sess:
        tf.saved_model.simple_save(sess,
                                   export_dir,
                                   inputs={'image_b':image_b,
                                           'max_outputs': max_outputs,
                                           'iou_thresh':iou_thresh,
                                           'score_thresh':score_thresh},
                                   outputs={'corners':pts_fin,
                                            'scores':scores_fin},
                                   legacy_init_op=tf.compat.v1.tables_initializer()
                                   )
