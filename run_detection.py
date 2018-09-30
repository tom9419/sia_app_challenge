import glob
import os
import time
import tensorflow as tf
import numpy as np
from shutil import copyfile
import cv2

def convert_index_to_category(idx):
    if idx == 1:
        category = 'Person'
    else:
        category = 'Misc'
    return category

path_to_graph = 'res/graph/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(path_to_graph, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

image_paths = []
path = 'res/'
for file_name in glob.glob(os.path.join(path, '*.jpg')):
    image_paths.append(file_name)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in image_paths:
      image = cv2.imread(image_path)
      image_np_expanded = np.expand_dims(image, axis=0)
      start = time.time()
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      end = time.time()
      print('Inference time:', end - start)
      boxes_np = np.squeeze(boxes)
      scores_np = np.squeeze(scores)
      classes_np = np.squeeze(classes)
      im_height, im_width, _ = image.shape
      for i in range(scores_np.shape[0]):
          if scores_np[i] > .5:
              category = convert_index_to_category(classes_np[i])
              if category == 'Person':
                  box = tuple(boxes_np[i].tolist())
                  ymin, xmin, ymax, xmax = box
                  cv2.rectangle(image, (int(xmin * im_width), int(ymin * im_height)),
                                (int(xmax * im_width), int(ymax * im_height)),
                                (255, 255, 0), 2)
      base_name = os.path.basename(image_path)
      image_det_path = path + '/result/' + os.path.splitext(base_name)[0] + '_det.jpg'
      cv2.imwrite(image_det_path, image)