import sys
import tensorflow as tf

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from pdb import set_trace as breakpoint


graph_pb = sys.argv[1]
export_dir = sys.argv[2]

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()

    ops = g.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type != 'Const':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    tensor = lambda x: g.get_tensor_by_name('%s:0' % x.name)
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {x.name:tensor(x) for x in inputs},
            {y.name:tensor(y) for y in outputs})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()
