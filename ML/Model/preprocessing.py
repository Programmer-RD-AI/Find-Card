from Model import *


class Pre_Processing:
    """sumary_line

    Keyword arguments:
    argument -- description
    Return: return_description
    """

    def __init__(self, init_compose: list = None) -> None:
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        if init_compose is None:
            init_compose = []
        self.compose_list = init_compose

    def center_crop(self, center_crop_amount):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.compose_list.append(torchvision.transforms.CenterCrop(center_crop_amount))
        return torchvision.transforms.CenterCrop(center_crop_amount)

    def color_jitter(self, brightness, contrast, saturation):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.compose_list.append(
            torchvision.transforms.ColorJitter(brightness, contrast, saturation)
        )
        return torchvision.transforms.CenterCrop(brightness, contrast, saturation)

    def random_grayscale(self, p):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.compose_list.append(torchvision.transforms.RandomGrayscale(p))
        return torchvision.transforms.RandomGrayscale(p)

    def random_horizontal_flip(self, p):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.compose_list.append(torchvision.transforms.RandomHorizontalFlip(p))
        return torchvision.transforms.RandomHorizontalFlip(p)

    def random_rotation(self, p):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.compose_list.append(torchvision.transforms.RandomHorizontalFlip(p))
        return torchvision.transforms.RandomHorizontalFlip(p)

    def random_vertical_flip(self, p):
        """sumary_line
        Keyword arguments:
        argument -- description
        Return: return_description
        """
        self.compose_list.append(torchvision.transforms.RandomVerticalFlip(p))
        return torchvision.transforms.RandomVerticalFlip(p)


# Ahhhhhhhhhhhhh I spend so much time but I forgot that there was something called torchvision.transforms What a waste of time anyways https://pytorch.org/vision/stable/transforms.html
