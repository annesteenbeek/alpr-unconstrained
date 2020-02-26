# Server side models

This folder holds code to build models that are compatible with the tensorflow serving infrastructure. Here we assume all models have been converted into the `.pb` format.

**Vehicle Det** + **Plate Det** + **OCR** = ***License Plate Recognition***

Test code are given in the `tests` folder. All testing scripts can be run under the project's root directory with no arguments, and demonstrative images will be generated so that we can have a subjective evaluation for the system.

## YOLOv3-SPP
A Darknet model. We use this as the *vehicle detector*.

Take the `416×416` model for example, there are 10647 boxes in the outputs.
Not every one of them are useful, so we use None Maximum Suppression to filter out those useless boxes. Let's download that to the `data/yolov3-spp` folder with this
```sh
cd data/yolov3-spp
wget https://pjreddie.com/media/files/yolov3-spp.weights
```

A object detection model trained with COCO dataset often predicts 80 categories of object. Plus the 2 coordinates, 2 shapes and 1 object score for each box. Let's prepare the class name file `coco.names` from [here](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

We use this [repo](https://github.com/mystic123/tensorflow-yolo-v3) to convert the Darknet model into TensorFlow model. Then build a serving model from that. A demo is provided with the repo, and we can verify that the conversion succeeds.

Upon successful conversion, we run the script `serverable/yolov3-spp.py` to build a server-side model.

```sh
python tools/serverable/yolov3-spp.py \
  data/yolov3-spp/frozen_darknet_yolov3_model.pb \
  data/yolov3-spp/1 \
  data/yolov3-spp/coco.names
```

Finally, we got the model ready for serving. The model directory looks like this:
```
data/yolov3-spp/
├── 1
│   ├── saved_model.pb
│   └── variables
├── coco.names
├── frozen_darknet_yolov3_model.pb
└── yolov3-spp.weights
```

## WPOD Net
A Keras model with tensorflow backend. TBD

## OCR Net
A Darknet model. TBD


# Integration
TBD


# Appendix
Useful snippets for building nested models in `GraphDef`

## load `.pb` models
```python
with tf.gfile.GFile(model_filepath, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
```

## assemble models
```python
with tf.Graph().as_default() as g_combined:
    x = tf.placeholder(tf.float32, name="")

    # Import gdef_1, which performs f(x).
    # "input:0" and "output:0" are the names of tensors in gdef_1.
    y, = tf.import_graph_def(gdef_1, input_map={"input:0": x},
                             return_elements=["output:0"])

    # Import gdef_2, which performs g(y)
    z, = tf.import_graph_def(gdef_2, input_map={"input:0": y},
                             return_elements=["output:0"]
```
