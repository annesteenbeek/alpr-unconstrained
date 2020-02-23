from src.keras_utils 			import load_model, detect_lp
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json

import sys

path = sys.argv[1]
export_path = sys.argv[2]
keras.backend.set_learning_phase(0)
# model = load_model(lp_model)

with open('%s.json' % path,'r') as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

with keras.backend.get_session() as sess:
    sess.run(tf.global_variables_initializer())
    model.load_weights('%s.h5' % path)
    tf.saved_model.simple_save(
          sess,
          export_path,
          inputs={'input_image': model.input},
          outputs={t.name: t for t in model.outputs}
    )
