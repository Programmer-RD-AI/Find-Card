import urllib.request

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except:
    raise ImportError(
        "Cannot Import Tqdm try installing it using `pip3 install tqdm` or `conda install tqdm`"
    )


class Download:
    def __init__(
        self,
        idx: int = 0,
        idx_1: int = 0,
        idx_2: int = 0,
        idx_3: int = 0,
        data: dict = {
            "ImageID": [],
            "OriginalURL": [],
            "OriginalLandingURL": [],
            "XMin": [],
            "YMin": [],
            "XMax": [],
            "YMax": [],
        },
        labels: list = [
            "Debit card",
            "Credit card",
            "Business card",
            "Collectible card game",
            "Telephone card",
            "Payment card",
        ],
        labels_r: list = [
            "/m/02h5d",
            "/m/0d7pp",
            "/m/01sdgj",
            "/m/0216z",
            "/m/066zr",
            "/m/09vh0m",
        ],
        labels_and_imageids: list = [
            "./open_images_data/validation-annotations-machine-imagelabels.csv",
            "./open_images_data/test-annotations-machine-imagelabels.csv",
            "./open_images_data/train-annotations-machine-imagelabels.csv",
            "./open_images_data/oidv6-train-annotations-human-imagelabels.csv",
            "./open_images_data/test-annotations-human-imagelabels-boxable.csv",
            "./open_images_data/validation-annotations-human-imagelabels-boxable.csv",
            "./open_images_data/train-annotations-human-imagelabels-boxable.csv",
        ],
        bboxs: list = [
            "./open_images_data/oidv6-train-annotations-bbox.csv",
            "./open_images_data/test-annotations-bbox.csv",
            "./open_images_data/validation-annotations-bbox.csv",
        ],
        image_urls: list = [
            "./open_images_data/oidv6-train-images-with-labels-with-rotation.csv",
            "./open_images_data/validation-images-with-rotation.csv",
            "./open_images_data/test-images-with-rotation.csv",
            "./open_images_data/train-images-boxable-with-rotation.csv",
        ],
        init_imageids: list = [],
        images_and_bbox_and_imgid_: list = [],
        imgids: list = [],
    ) -> None:
        """summary_line

        Keyword arguments:
        argument
            idx,idx_1,idx_2,idx_3 = the init starting index
            data = The data that is going to be download if want to add new info can be done.
            labels = labels from open images (Name of label)
            labels_r = labels from open images (Label Code of label)
            labels_and_imageids = loading files for getting the labels_and_imageids
            bboxs = loading files for getting the bboxs
            image_urls = loading files for getting the image_urls
            init_imageids = init imageids
            images_and_bbox_and_imgid_ = init images_and_bbox_and_imgid_
            init imgids = init imgids
        Return: None
        """
        try:
            # Indexing
            self.idx = idx
            self.idx_1 = idx_1
            self.idx_2 = idx_2
            self.idx_3 = idx_3
        except Exception as e:
            raise ValueError(
                f"In the indexing parameters there was a error occurred {e}")
        try:
            # Initinalizing Data Storage
            self.data = data
        except Exception as e:
            raise ValueError(
                f"In the Initinalizing Data Storage parameters there was a error occurred {e}"
            )
        try:
            # Labels of data
            self.labels = labels
            self.labels_r = labels_r
        except Exception as e:
            raise ValueError(
                f"In the Labels of data parameters there was a error occurred {e}"
            )
        try:
            # Data Loading file paths
            self.labels_and_imageids = labels_and_imageids
            self.bboxs = bboxs
            self.image_urls = image_urls
        except Exception as e:
            raise ValueError(
                f"In the Data Loading file paths parameters there was a error occurred {e}"
            )
        try:
            # Collection of data
            self.imageids = init_imageids
            self.images_and_bbox_and_imgid_ = images_and_bbox_and_imgid_
            self.imgids = imgids
        except Exception as e:
            raise ValueError(
                f"In the Collection of data parameters there was a error occurred {e}"
            )

    ## Loading data section ##

    def load_labels_and_imageid(self) -> pd.DataFrame:
        """summary_line

        Keyword arguments:
        argument load_labels_and_imageid
        Return: pd.DataFrame
        """
        try:
            labels_and_imageid = pd.read_csv(self.labels_and_imageids[0])
            for i in range(1, len(self.labels_and_imageids)):
                labels_and_imageid.append(
                    pd.read_csv(self.labels_and_imageids[i]))
            return labels_and_imageid
        except Exception as e:
            raise ValueError(
                f"The function self.load_labels_and_imageid() or Download().load_labels_and_imageid() is not working correctly. {e}"
            )

    def load_bbox(self) -> pd.DataFrame:
        """summary_line

        Keyword arguments:
        argument load_bbox
        Return: pd.DataFrame
        """
        try:
            bboxs_df = pd.read_csv(self.bboxs[0])
            for i in range(1, len(self.bboxs)):
                bboxs_df.append(pd.read_csv(self.bboxs[i]))
            return bboxs_df
        except Exception as e:
            raise ValueError(
                f"The function self.load_bbox() or Download().load_bbox() is not working correctly. {e}"
            )

    def load_image_urls(self) -> pd.DataFrame:
        """summary_line

        Keyword arguments:
        argument load_image_urls
        Return: pd.DataFrame
        """
        try:
            image_urls_df = pd.read_csv(self.image_urls[0])
            for i in range(1, len(self.image_urls)):
                image_urls_df.append(pd.read_csv(self.image_urls[i]))
            return image_urls_df
        except Exception as e:
            raise ValueError(
                f"The function self.load_image_urls() or Download().load_image_urls() is not working correctly. {e}"
            )

    ## Creating data section ##

    def create_imageids(self) -> bool:
        """summary_line

        Keyword arguments:
        argument create_imageids
        Return: bool
        """
        try:
            labels_and_imageid = self.load_labels_and_imageid()
            print(len(labels_and_imageid))
            for labelname, imageid in tqdm(
                    zip(labels_and_imageid["LabelName"],
                        labels_and_imageid["ImageID"])):
                if labelname in self.labels_r:
                    self.idx_1 += 1
                    self.imageids.append(imageid)
            del labels_and_imageid
            return True
        except Exception as e:
            raise ValueError(
                f"The function self.create_imageids() or Download().create_imageids() is not working correctly. {e}"
            )

    def create_bbox(self) -> bool:
        """summary_line

        Keyword arguments:
        argument create_bbox
        Return: bool
        """
        try:
            bboxs = self.load_bbox()
            print(len(bboxs))
            for imgid in tqdm(
                    zip(
                        bboxs["ImageID"],
                        bboxs["XMin"],
                        bboxs["YMin"],
                        bboxs["XMax"],
                        bboxs["YMax"],
                    )):
                if imgid[0] in self.imageids:
                    self.idx_2 += 1
                    self.images_and_bbox_and_imgid_.append(imgid)
                    self.imgids.append(imgid[0])
            np.save("./imageids.npy", self.imgids)
            del bboxs
            self.images_and_bbox_and_imgid_ = pd.DataFrame(
                self.images_and_bbox_and_imgid_,
                columns=["ImageID", "XMin", "YMin", "XMax", "YMax"],
            )
            print(len(self.images_and_bbox_and_imgid_))
            return True
        except Exception as e:
            raise ValueError(
                f"The function self.create_bbox() or Download().create_bbox() is not working correctly. {e}"
            )

    def create_image_urls(self) -> bool:
        """summary_line

        Keyword arguments:
        argument create_image_urls
        Return: bool
        """
        try:
            data = {
                "ImageID": [],
                "OriginalURL": [],
                "OriginalLandingURL": [],
                "XMin": [],
                "YMin": [],
                "XMax": [],
                "YMax": [],
            }
            image_urls = self.load_image_urls()
            print(len(image_urls))
            for imgid in tqdm(
                    zip(
                        image_urls["ImageID"],
                        image_urls["OriginalURL"],
                        image_urls["OriginalLandingURL"],
                    )):
                if imgid[0] in self.imgids:
                    imgid_of_iabaid = self.images_and_bbox_and_imgid_[
                        self.images_and_bbox_and_imgid_["ImageID"] == imgid[0]]
                    for idx_3 in range(len(imgid_of_iabaid)):
                        imgid_of_iabaid_iter = self.images_and_bbox_and_imgid_[
                            self.images_and_bbox_and_imgid_["ImageID"] ==
                            imgid[0]].iloc[idx_3]
                        data["ImageID"].append(imgid[0])
                        data["OriginalURL"].append(imgid[1])
                        data["OriginalLandingURL"].append(imgid[2])
                        data["XMin"].append(imgid_of_iabaid_iter["XMin"])
                        data["YMin"].append(imgid_of_iabaid_iter["YMin"])
                        data["XMax"].append(imgid_of_iabaid_iter["XMax"])
                        data["YMax"].append(imgid_of_iabaid_iter["YMax"])
            del self.images_and_bbox_and_imgid_
            data = pd.DataFrame(data)
            self.download_url_data = data
        except Exception as e:
            raise ValueError(
                f"The function self.create_image_urls() or Download().create_image_urls() is not working correctly. {e}"
            )

    ## Downloading data section ##

    def download_images(self) -> pd.DataFrame:
        """summary_line

        Keyword arguments:
        argument download images
        Return: pd.DataFrame
        """
        try:
            new_data = {
                "Path": [],
                "XMin": [],
                "YMin": [],
                "XMax": [],
                "YMax": [],
                "ImageID": [],
            }
            for img_url, xmin, ymin, xmax, ymax, ourl in tqdm(
                    zip(
                        self.download_url_data["ImageID"],
                        self.download_url_data["XMin"],
                        self.download_url_data["YMin"],
                        self.download_url_data["XMax"],
                        self.download_url_data["YMax"],
                        self.download_url_data["OriginalURL"],
                    )):
                try:
                    self.idx += 1
                    urllib.request.urlretrieve(ourl, f"./Img/{self.idx}.png")
                    new_data["Path"].append(f"{self.idx}.png")
                    new_data["XMin"].append(xmin)
                    new_data["YMin"].append(ymin)
                    new_data["XMax"].append(xmax)
                    new_data["YMax"].append(ymax)
                    new_data["Url"].append(ourl)
                    new_data["ImageID"].append(img_url)
                except Exception as e:
                    pass
            print(len(new_data))
            data = pd.DataFrame(new_data)
            data.to_csv("./Data.csv", index=False)
            return new_data
        except Exception as e:
            raise ValueError(
                f"The function self.download_images() or Download().download_images() is not working correctly. {e}"
            )

    ## Download.download() ##

    def download(self) -> bool:
        """summary_line

        Keyword arguments:
        argument: This is the funtion which uses
            all of the funtions of this class
            and combines it and download and
            does all of the work.
        Return: bool
        """
        try:
            self.create_imageids()
            self.create_bbox()
            self.create_image_urls()
            return self.download_images()
        except Exception as e:
            raise ValueError(
                f"The function self.download() or Download().download() is not working correctly. {e}"
            )


d = Download()
d.download()
