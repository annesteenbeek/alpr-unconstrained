#!/bin/bash
set -e

if [[ "$VIRTUAL_ENV" == "" ]]; then
  echo ""
  read -p "Not in python virtual env, continue? (y/n)" -n 1 -r
  echo    # (optional) move to a new line
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Exiting"
    exit 1
  fi
fi

# get root dir where script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# make sure cwd is root dir
cd "$DIR"

if [ ! -d "data/lp-detector" ]; then
  bash get-networks.sh
fi

# ----- YOLOv3-SPP ----- (vehicle detection)
echo "Setting up YOLOv3-SPP"
if [ ! -d "tensorflow-yolo-v3" ]; then
  git clone https://github.com/mystic123/tensorflow-yolo-v3.git
fi

if [ -f "data/yolov3-spp/yolov3-spp.weights" ]; then
  echo "weights already exists, skipping"
else
  mkdir -p data/yolov3-spp
  echo "Getting yolov3 weights"
  wget https://pjreddie.com/media/files/yolov3-spp.weights -P data/yolov3-spp
  wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -P data/yolov3-spp
fi

if [ ! -d "data/yolov3-spp/1" ]; then
  echo "Converting yolov3 weights to tensorflow .pb"
  cd data/yolov3-spp
  python ../../tensorflow-yolo-v3/convert_weights_pb.py \
    --class_names coco.names \
    --data_format NHWC \
    --weights_file yolov3-spp.weights \
    --spp 

  cd "$DIR"
  # build server side model
  python tools/servable/yolov3-spp.py \
    data/yolov3-spp/frozen_darknet_yolov3_model.pb \
    data/yolov3-spp/1 \
    data/yolov3-spp/coco.names
fi

# ---- WPOD-Net ------ (License plate detection)
if [ ! -d "data/lp-detector/1" ]; then
  python tools/servable/wpod-net.py \
    data/lp-detector/wpod-net_update1 \
    data/lp-detector/1
fi

# ---- OCR-Net ------- (OCR)
# install darkflow 
if [ ! -d "darkflow" ]; then
  git clone https://github.com/thtrieu/darkflow.git
  cd darkflow
  git apply ../darkflow_offset.patch
  python setup.py build_ext --inplace
  cd ..
fi

if [ ! -d "data/ocr/1" ]; then
  cd data/
  ./../darkflow/flow \
    --model ocr/ocr-net.cfg \
    --load ocr/ocr-net.weights \
    --labels ocr/ocr-net.names \
    --savepb

  # Convert ocr-net to tensorflow servable
  cd "$DIR"
  python tools/servable/ocr-net.py \
    data/built_graph/ocr-net.pb \
    data/ocr/1 \
    data/ocr/ocr-net.names
fi