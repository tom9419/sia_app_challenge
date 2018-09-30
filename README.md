# Person counting for waiting queues
This is a Project for the Singapore Airlines AppChallenge 2018, in which a person counter for waiting queues is developed.
Therefore a Person detection is used, using the Tensorflow Object Detction API.
The person detection is to be extended by a person tracking.

## Setup
### 1. Install Tensorfow and OpenCV
```
pip3 install tensorflow
```
or
```
pip3 install tensorflow-gpu
```
```
pip3 install opencv-python
```
### 2. Clone the repository
```
git clone https://github.com/tom9419/sia_app_challenge.git
```
### 3. Download the graph
Download the graph from the Tensorflow Object Detection API trained on the COCO dataset [here](http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_2018_01_28.tar.gz).
Extract it and place the frozen_inference_graph.pb in the `/res/graph/` folder.

## Run the graph
Run the graph for the example images in `/res/` and write the results to `/res/results`
```
cd sia_app_challenge
python3 run_detection.py 
```



