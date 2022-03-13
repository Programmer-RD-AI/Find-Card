"""sumary_line

Keyword arguments:
argument -- description
Return: return_description
"""
import os

from imageai.Detection import ObjectDetection

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(
    os.path.join(execution_path, "./testing/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(
    person=True,
    car=False,
)
detections = detector.detectCustomObjectsFromImage(
    input_image=os.path.join(execution_path, "image.png"),
    output_image_path=os.path.join(execution_path, "image_new.png"),
    custom_objects=custom_objects,
    minimum_percentage_probability=65,
)
