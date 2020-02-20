# python 2
cd `dirname $0`/..
conda install tensorflow-gpu==1.15.0
pip install numpy==1.14 keras==2.2.4 opencv-python==3.4.9.31
cd darknet && make GPU=1 && cd ..
bash get-networks.sh
