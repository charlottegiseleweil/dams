apt-get update -y
apt-get install -y git wget python3-tk

pip install --upgrade pip
pip install tqdm Cython contextlib2 pillow lxml jupyter matplotlib

cd /lib/tf ## change to INSTALL_DIR !

git clone https://github.com/tensorflow/models
cd models
git reset --hard 8367cf6dabe11adf7628541706b660821f397dce
cd .. 

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools ../../models/research/
cd ../..

mkdir protoc_3.3
cd protoc_3.3
wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip
chmod 775 protoc-3.3.0-linux-x86_64.zip
unzip protoc-3.3.0-linux-x86_64.zip

cd ../models/research

apt-get install -y protobuf-compiler

echo *** Installed protobuf-compiler

../../protoc_3.3/bin/protoc object_detection/protos/*.proto --python_out=.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python object_detection/builders/model_builder_test.py
echo **** PYTHONPATH used to run model_builder_test.py
echo $PYTHONPATH

python setup.py sdist
(cd slim && python setup.py sdist)

echo *********** PWD is
echo $PWD
echo *****
