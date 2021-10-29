from Model import *
import unittest

model = Model()


class Test_Model(unittest.TestCase):
    def test_remove_files_in_output(self):
        model.remove_files_in_output()
        self.assertEqual(len(os.listdir("./output/")), 0, "Should be 0")

    def test_predict_test_imgs(self):

        self.assertEqual(type(model.test()[0]), np.ndarray)
        self.assertEqual(type(model.test()[1]), np.ndarray)

    def test_load_data(self):

        self.assertEqual(len(model.load_data()), len(model.data))


if __name__ == "__main__":
    unittest.main()
