import datetime
import os

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tflite_model_maker import model_spec, object_detector
from tflite_model_maker.config import ExportFormat, QuantizationConfig

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices("GPU")))


class TFLite:
    """sumary_line

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def __init__(
        self,
        model: str = "efficientdet_lite2",
        batch_size: int = 16,
        epochs: int = 100,
        save_file_name: str = "model.tflite",
        train_whole_model: bool = True,
        export_dir: str = ".",
        model_path: str = "model.tflite",
        data_loading_csv: str = "gs://cloud-ml-data/img/openimage/csv/salads_ml_use.csv",
    ) -> None:
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.model = model
        self.batch_size = batch_size
        self.save_file_name = save_file_name
        self.train_whole_model = train_whole_model
        self.export_dir = export_dir
        self.data_loading_csv = data_loading_csv
        self.log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.epochs = epochs
        self.threshold = 0.125
        self.model_path = model_path

    def load_data(self) -> tuple:
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        with tf.device("/GPU:0"):
            print("Loading data")
            (
                self.train_data,
                self.validation_data,
                self.test_data,
            ) = object_detector.DataLoader.from_csv(self.data_loading_csv)
            return self.train_data, self.validation_data, self.test_data

    def create_model(self) -> tuple:
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        with tf.device("/GPU:0"):
            print("Creating Model")
            self.spec = model_spec.get(self.model)
            return self.spec

    def train_model(self) -> tuple:
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        with tf.device("/GPU:0"):
            print("Train Model")
            self.model = object_detector.create(
                self.train_data,
                model_spec=self.spec,
                batch_size=self.batch_size,
                train_whole_model=self.train_whole_model,
                validation_data=self.validation_data,
                epochs=self.epochs,
            )
            return self.model

    def evaluate(self) -> tuple:
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        with tf.device("/GPU:0"):
            print("Evaluate")
            return self.model.evaluate(self.test_data)

    def save(self) -> tuple:
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        with tf.device("/GPU:0"):
            print("Save")
            self.model.export(export_dir=self.export_dir)
            self.model.evaluate_tflite(self.save_file_name, self.test_data)
            return ()

    def load(self):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        with tf.device("/GPU:0"):
            print("Load")
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.predictor = self.interpreter.get_signature_runner()
            return self.predictor, self.interpreter

    def create_test_image(self, image_path):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        with tf.device("/GPU:0"):
            print("Creating test images")
            _, input_height, input_width, _ = self.interpreter.get_input_details()[0]["shape"]
            # Read image in tf encoded format
            img = tf.io.read_file(image_path)
            img = tf.io.decode_image(img, channels=3)  # Decode the image (Load the image)
            img = tf.image.convert_image_dtype(img, tf.uint8)  # Convert to Data Type Unit8
            self.original_image = img
            self.resized_img = tf.image.resize(img, (input_height, input_width))  # Resize Image
            self.resized_img = self.resized_img[tf.newaxis, :]  # Add 1 dimension to the image
            self.resized_img = tf.cast(
                self.resized_img, dtype=tf.uint8
            )  # Convert to Data Type Unit8

    def predict_test_image(self):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        with tf.device("/GPU:0"):
            print("Predict Test Image")
            preds = self.predictor(images=self.resized_img)
            count = int(np.squeeze(preds["output_0"]))
            scores = np.squeeze(preds["output_1"])
            classes = np.squeeze(preds["output_2"])
            boxes = np.squeeze(preds["output_3"])
            results = []
            for i in range(count):
                if scores[i] >= self.threshold:
                    result = {
                        "bounding_box": boxes[i],
                        "class_id": classes[i],
                        "score": scores[i],
                    }
                    results.append(result)
            return results


tflite = TFLite()
tflite.load_data()
tflite.create_model()
tflite.train_model()
tflite.evaluate()
tflite.save()
tflite.load()
tflite.create_test_image()
tflite.predict_test_image()
