# Install
`bash install.sh`


# Building an Inference Server(WIP)

Darknet uses images files as network input, which is slow. In addition, there'are some difficulties when using the default models in Darknet on GPU's.

As a walk around, we convert *Darknet* models into the *TensorFlow* format.
The details are in the `servable` subfolder.


# Getting Structured LPR Data from Videos and Images
Vehicles' location, plates' location and license numbers are acquired when processing the raw data with the LPR model. We use this information to assist data annotation.

## preparation
- If there's any video, extract frames from it.

```bash
bash video_batch.sh ${YOUR_VIDEO_DIR}
```

- Rotate images if necessary
```bash
for image in `ls`; do
  ffmpeg -y -i ${image} -vf "rotate=PI" ${image}
done
```

- Make sure images' names are unique.

```bash
for i in `ls`;do
  for j in `ls $i`;do
    if [[ ${j} != *"IMG"* ]];then
      mv "$i/$j" "$i/${i}_$j";
    fi
  done;
  echo "$i done"
done
```

## run LPR

An Image folder has subfolders that hold images.

```bash
bash batch_lpr.sh ${YOUR_IMAGE_FOLDER}
```


# Annotation

## set up coco annotator
1. `docker-compose -f docker-compose.dev.yml up`
2. open the web client, register a user, write a `login.ini` configuration file like the following:
  >[coco-annotator]
  address = localhost
  port = 8080
  username = hibike
  password = euphonium

3. create a dataset `${your_dataset}`
4. move some images to the `./datasets/${your_dataset}`

<!-- # remove database from coco annotator
`docker-compose down`
`docker volume prune` -->
## generate annotation
```bash
python txt2json.py ${login.ini} ${txt_dir} ${dataset_id}
# wait until all tasks are finished in the web client
python txt2json.py ${login.ini} ${txt_dir} ${dataset_id}
# twice to ensure liscense plates are uploaded to the DB
```
download the JSON file after manual check

## build data sets

```bash
python tools/json2dataset.py \
  ${ANNOTATION_FILE_IN_JSON} \
  ${IMAGE_FOLDER} \
  ${export_dir}
```


# Training Models(WIP)

## extract LP data for training
extract LP localization from the JSON file
in the folder `samples/train-detector`

## LP detector
  We use `docker` to save us from configuring GPU's to work with TensorFlow

  1. enter in the docker container
    ```bash
    docker run -it \
      --runtime=nvidia \
      -v $(pwd):/workspace \
      -v ${PATH_TO_YOUR_TRAINING}/data \
      tensorflow/tensorflow:1.15.2-gpu \
      bash
    ```

  2. go to the project root folder mounted by the container
    `cd workspace`

  3. start training with the following command
    ```bash
    python train-detector.py \
      --model data/lp-detector \
      --name my-trained-model \
      --train-dir /data \
      --output-dir models/my-trained-model/
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
