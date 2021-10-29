from collections import OrderedDict
from Model import *
import unittest

model = Model()


class Test_Model(unittest.TestCase):
    def test_predict_test_imgs(self):

        self.assertEqual(
            type(model.test()[0]), np.ndarray, "The Crop and Box Image is None"
        )
        self.assertEqual(
            type(model.test()[1]), np.ndarray, "The Crop and Box Image is None"
        )

    def test_load_data(self):
        self.assertEqual(
            len(model.load_data()),
            len(model.data),
            "The length is not correct of the records and the dataframe",
        )

    def test_save(self):
        model.save(test="test")
        if f"test-{model.NAME}.pt" in os.listdir("./models/"):
            self.assertEqual(True, True)
        else:
            self.assertEqual(False, True)

    def test_create_cfg(self):
        self.assertEqual(type(model.create_cfg()), CfgNode)

    def test_create_predictor(self):
        self.assertEqual(type(model.create_predictor()), DefaultPredictor)

    def test_create_coco_eval(self):
        self.assertEqual(
            type(model.create_coco_eval(model.create_predictor())), OrderedDict
        )

    def test_metrics_file_to_dict(self):
        self.assertEqual(type(model.metrics_file_to_dict()), list)

    def test_predict_test_images(self):
        self.assertEqual(
            type(model.predict_test_images(model.create_predictor())), list
        )

    def test_create_target_and_preds(self):
        self.assertEqual(
            type(model.create_target_and_preds(model.create_predictor())), tuple
        )

    def test_create_rmse(self):
        (
            preds,
            target,
            x,
            y,
            w,
            h,
            xmin,
            ymin,
            xmax,
            ymax,
            height,
            width,
        ) = model.create_target_and_preds(model.create_predictor())
        self.assertEqual(type(model.create_rmse(preds, target)), float)

    def test_create_mse(self):
        (
            preds,
            target,
            x,
            y,
            w,
            h,
            xmin,
            ymin,
            xmax,
            ymax,
            height,
            width,
        ) = model.create_target_and_preds(model.create_predictor())
        self.assertEqual(type(model.create_mse(preds, target)), float)

    def test_create_x_y_w_h(self):
        self.assertEqual(type(model.create_x_y_w_h(59, 59, 89, 59)), list)

    def test_remove_files_in_output(self):
        model.remove_files_in_output()
        self.assertEqual(len(os.listdir("./output/")), 0, "Should be 0")


if __name__ == "__main__":
    unittest.main()
