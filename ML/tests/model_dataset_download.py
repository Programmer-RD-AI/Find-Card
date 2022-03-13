import unittest

import pandas as pd

from Model.dataset.download import *


class Test_Model_DataSet_Download(unittest.TestCase):
    """sumary_line

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def __init__(self):
        """sumary_line

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.labels_and_imageids = [
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/oidv6-train-annotations-human-imagelabels.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/test-annotations-human-imagelabels-boxable.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/test-annotations-machine-imagelabels.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/train-annotations-human-imagelabels-boxable.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/train-annotations-machine-imagelabels.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/validation-annotations-human-imagelabels-boxable.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/validation-annotations-machine-imagelabels.csv",
        ]
        self.bboxs = [
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/oidv6-train-annotations-bbox.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/test-annotations-bbox.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/validation-annotations-bbox.csv",
        ]
        self.image_urls = [
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/oidv6-train-images-with-labels-with-rotation.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/test-images-with-rotation.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/train-images-boxable-with-rotation.csv",
            "/media/indika/Sync/Programmer-RD-AI/Programming/Projects/Python/Rest-Api/Car-Object-Detection-REST-API/Find-Card/Model/dataset/open_image_raw_data/validation-images-with-rotation.csv",
        ]
        self.download = Download(labels=["/m/019dxh"], labels_r=["/m/019dxh"])

    def test_load_labels_and_imageid(self):
        """sumary_line

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        gc.collect()
        labels_and_imageid = pd.read_csv(self.labels_and_imageids[0])
        loader_iter = range(1, len(self.labels_and_imageids))
        for i in loader_iter:
            labels_and_imageid = labels_and_imageid.append(
                pd.read_csv(self.labels_and_imageids[i]))
        labels_and_imageid.sample(frac=1)
        gc.collect()
        self.assertEqual(
            len(labels_and_imageid),
            len(self.download.load_labels_and_imageid()),
            "The Length Collection of Labels and Imageids csv files Are not equal",
        )

    def test_load_bbox(self):
        """sumary_line

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        gc.collect()
        bboxs_df = pd.read_csv(self.bboxs[0])
        loader_iter = range(1, len(self.bboxs))
        for i in loader_iter:
            loader_iter.set_description(str(len(bboxs_df)))
            bboxs_df = bboxs_df.append(pd.read_csv(self.bboxs[i]))
        bboxs_df.sample(frac=1)
        gc.collect()
        self.assertEqual(
            len(bboxs_df),
            len(self.download.load_bbox()),
            "The Length Collection of Bbox csv files Are not equal",
        )

    def test_load_image_urls(self):
        """sumary_line

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        gc.collect()
        image_urls_df = pd.read_csv(self.image_urls[0])
        loader_iter = tqdm(range(1, len(self.image_urls)))
        for i in loader_iter:
            loader_iter.set_description(str(len(image_urls_df)))
            image_urls_df = image_urls_df.append(
                pd.read_csv(self.image_urls[i]))
        image_urls_df.sample(frac=1)
        gc.collect()
        self.assertEqual(
            len(image_urls_df),
            len(self.download.load_image_urls()),
            "The Length Collection of Image Urls csv files Are not equal",
        )

    def test_create_imageids(self):
        """sumary_line
        Keyword arguments:
            argument -- description
        Return: return_description
        """
        self.assertEqual(self.download.create_imageids(), True,
                         "ImageIds wasnt created successfully")

    def test_create_bbox(self):
        """sumary_line

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.assertEqual(self.download.create_bbox(), True,
                         "Bbox wasnt created successfully")

    def test_create_image_urls(self):
        """sumary_line

        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.assertEqual(
            self.download.create_image_urls(),
            True,
            "Image Urls wasnt created successfully",
        )


if __name__ == "__main__":
    unittest.main()
