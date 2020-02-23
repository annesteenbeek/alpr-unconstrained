# install
`bash install.sh`

# extract frames

## darknet uses images files as network input
as a walk around, we convert darknet to tensorflow

## freeze darknet as tensorflow pb
use this [repo](https://github.com/mystic123/tensorflow-yolo-v3)
`python convert_weights_pb.py --class_names data/yolov3-spp/coco.names --data_format NHWC --weights_file data/yolov3-spp/yolov3-spp.weights --spp`

## build saved models
`python tools/pb2savedmodel.py data/yolov3-spp/frozen_darknet_yolov3_model.pb data/yolov3-spp/1`
### view saved model
`saved_model_cli show --dir data/yolov3-spp/1 --all`
> MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 416, 416, 3)
        name: inputs:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['output_boxes'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 10647, 85)
        name: output_boxes:0
  Method name is: tensorflow/serving/predict

## DarkFlow

## tensorflow serving
`docker-compose up`

```bash
bash video_batch.sh ${YOUR_VIDEO_PATH}
```
# run LPR
```bash
bash batch_lpr.sh ${YOUR_IMAGE_FOLDER}
```
# flatten nested folders
perform once for the images folder;
and the other for the text file folder
```bash
for i in `ls`;do
  for j in `ls $i`;do
    mv $i/$j ${i}_$j;
  done;
  rm -d $i
done
```
# set up coco annotator
`docker-compose -f docker-compose.dev.yml up`
create a dataset `lpr0218`
move images to the ./datasets/lpr0218

<!-- # remove database from coco annotator
`docker-compose down`
`docker volume prune` -->
# generate annotation
```bash
python txt2json.py
python txt2json.py # twice to ensure liscense plates are uploaded to the DB
```
download the JSON file after manual check

# get applicable images
```bash
python tools/useful_images.py ${YOUR_ANNOTATION_JSON_FILE}
```
# extract LP data for training
extract LP localization from the JSON file
in the folder `samples/train-detector`
## LP detector
  ```bash
  python train-detector.py \
    --model data/lp-detector \
    --name my-trained-model \
    --train-dir 'samples/train-detector' \
    --output-dir models/my-trained-model/ \
    -op Adam \
    -lr .001 \
    -its 300000 \
    -bs 64
  ```
## LP recognizer
### prepare training data
  tackle OCR as a detection task
  - image list and folder
  - label folder
  >train=
  >valid=  

### train
  ```bash
  ./darknet/darknet detector train \
    data/ocr/ocr-net.data \
    data/ocr/ocr-net.cfg \
    ocr-net.weights
  ```
