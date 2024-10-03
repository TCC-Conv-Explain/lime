from numpy.typing import NDArray
from torch import Tensor

from skimage.io import imread, imshow
from skimage.transform import resize

from torchvision.transforms import ToTensor

from numpy import float32

def read_image(image_path: str, size: tuple[int]=None):
    image = imread(image_path)
    if size is not None:
        image = resize(image, size)
    
    return image

def numpy_to_torch(image: NDArray) -> Tensor:
    to_tensor = ToTensor()
    tensor_image = to_tensor(image.astype(float32))
    return tensor_image.unsqueeze(0)