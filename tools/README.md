# install
`bash install.sh`

# extract frames

## darknet uses images files as network input
as a walk around, we convert darknet to tensorflow

## freeze darknet as tensorflow pb
use this [repo](https://github.com/mystic123/tensorflow-yolo-v3)
`python convert_weights_pb.py --class_names data/yolov3-spp/coco.names --data_format NHWC --weights_file data/yolov3-spp/yolov3-spp.weights --spp`

## build saved models
`python tools/pb2savedmodel.py data/yolov3-spp/frozen_darknet_yolov3_model.pb data/yolov3-spp-savedmodel/1`
### view saved model
`saved_model_cli show --dir data/yolov3-spp-savedmodel/1 --all`
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
# generate annotation
```bash
python txt2json.py
```
# get applicable images
```bash
python tools/useful_images.py ${YOUR_ANNOTATION_JSON_FILE}
```
# aggregate useful images
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
