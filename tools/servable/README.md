# Server-side Models

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

We use this [repository](https://github.com/mystic123/tensorflow-yolo-v3) to convert the Darknet model into TensorFlow model. Then build a serving model from that. A demo is provided with the repository, and we can verify that the conversion succeeds.

```bash
python convert_weights_pb.py \
  --class_names coco.names \
  --data_format NHWC \
  --weights_file yolov3-spp.weights \
  --spp
```

Upon successful conversion, we run the script `servable/yolov3-spp.py` to build a server-side model.

```sh
python tools/servable/yolov3-spp.py \
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

## WPOD-Net
A Keras model with TensorFlow backend. We use the following script to build serving models from the ones in Keras.

```bash
  python tools/servable/wpod-net.py \
    data/lp-detector/wpod-net_update1 \
    data/lp-detector/1
```

The models should look like:
```
data/lp-detector/
├── 1
│   ├── saved_model.pb
│   └── variables
├── wpod-net_update1.h5
└── wpod-net_update1.json
```

Use the CLI command to inspect the inputs and outputs:
`saved_model_cli show --dir data/lp-detector/1 --all`

## OCR-Net
A Darknet model.

1. install [darkflow](https://github.com/thtrieu/darkflow) and convert the ocr net to tensorflow model in `.pb` format.
  ```bash
  ./flow \
    --model ocr/ocr-net.cfg \
    --load ocr/ocr-net.weights \
    --labels ocr/ocr-net.names \
    --savepb
  ```

2. convert the graph model to the format of serving with the script in the `servable` folder
  ```bash
  python tools/servable/ocr-net.py \
    darkflow/built_graph/ocr-net.pb \
    data/ocr/1 \
    data/ocr/ocr-net.names
  ```

The models should look like:
```
data/ocr/
├── 1
│   ├── saved_model.pb
│   └── variables
├── ocr-net.cfg
├── ocr-net.data
├── ocr-net.names
└── ocr-net.weights
```


# Launch the service
use the command:
`docker-compose up`

Then we can run the scripts in the `test` folder to see the performance of each serving model.
