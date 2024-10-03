import numpy as np

from numpy.random import binomial, seed
from numpy.typing import NDArray

from skimage.io import imread, imsave
from skimage.segmentation import quickshift

from torch import argmax
from torch import Tensor
from torch.nn import Module

from torchvision.models import vgg16, VGG16_Weights

from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances

from PIL import Image

from tqdm import tqdm

import argparse

from utils import *

def gen_image_superpixels(image: NDArray, kernel_size: int, max_dist: int, ratio: float) -> NDArray:
    return quickshift(image, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)

def get_superpixels_num(superpixels: NDArray) -> int:
    return len(set(superpixels.flatten().tolist()))

def gen_superpixels_sample(num_samples: int, num_superpixels: int, probability:float, random_seed: int = None) -> NDArray:
    if random_seed is not None:
        seed(random_seed)
    return binomial(1, probability, (num_samples, num_superpixels))

def gen_masked_image(image: NDArray, superpixels_sample: NDArray, image_superpixels: NDArray):
    sample_indexes = np.where(superpixels_sample == 1)
    mask = np.isin(image_superpixels, sample_indexes)
    return image * mask[..., None]  # clever way to multiply (W, H, C) * (W, H)  :D

def gen_model_preds(model: Module, image: NDArray, superpixels_sample: NDArray, image_superpixels: NDArray) -> Tensor:
    masked_image = gen_masked_image(image, superpixels_sample, image_superpixels)
    tensor_image = numpy_to_torch(masked_image)
    return model(tensor_image)

def find_class_to_explain(model: Module, image: NDArray, image_superpixels: NDArray) -> Tensor:
    superpixels_num = get_superpixels_num(image_superpixels)
    pred = gen_model_preds(model, image, np.ones((superpixels_num, )), image_superpixels)
    return argmax(pred)

def get_sample_distances(superpixels_sample: NDArray) -> NDArray:
    complete_sample = np.ones_like(superpixels_sample)[0]
    complete_sample = np.expand_dims(complete_sample, axis=0)
    return np.squeeze(pairwise_distances(superpixels_sample, complete_sample, metric='cosine'))

def get_sample_weights(image_superpixels: NDArray, kernel_width: float=0.25) -> NDArray:
    sample_distances = get_sample_distances(image_superpixels)
    return np.sqrt(np.exp(- sample_distances**2 / kernel_width**2))

def train_linear_model(
    image: NDArray,
    model: Module,
    image_superpixels: NDArray,
    seed: int,
    sampling_prob: float,
    sampling_num: int,
    distance_kernel: float
    ):
    superpixels_num = get_superpixels_num(image_superpixels)
    class_to_explain = find_class_to_explain(model, image, image_superpixels)

    superpixels_sample = gen_superpixels_sample(sampling_num, superpixels_num, sampling_prob, random_seed=seed)

    outs = []
    for sample in tqdm(superpixels_sample):
        pred = gen_model_preds(model, image, sample, image_superpixels)
        outs.append(pred[0][class_to_explain].detach())

    linear_model = LinearRegression()
    linear_model.fit(X=superpixels_sample, y=outs, sample_weight=get_sample_weights(superpixels_sample, distance_kernel))
    
    return linear_model.coef_

def lime(
    path: str,
    seed: int,
    sampling_prob: float,
    sampling_num: int,
    quickshift_kernel: int,
    quickshift_max_dist: int,
    quickshift_ratio: float,
    distance_kernel: float,
    num_selected_coefs: int
    ) -> NDArray:

    image = imread(path)
    image_superpixels = gen_image_superpixels(image, quickshift_kernel, quickshift_max_dist, quickshift_ratio)
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    
    coefs = train_linear_model(image, model, image_superpixels, seed, sampling_prob, sampling_num, distance_kernel)
    
    top_features = np.argsort(coefs)[-num_selected_coefs:]

    mask = np.zeros(get_superpixels_num(image_superpixels))
    mask[top_features] = 1

    return gen_masked_image(image, mask, image_superpixels)

def main():
    parser = argparse.ArgumentParser(description="DeepDream in PyTorch using VGG16")

    parser.add_argument("path", type=str, help="Path to image")
    
    parser.add_argument("--seed", type=int, default=42, help="Random seed value")
    parser.add_argument("--sampling-prob", type=float, default=0.5, help="Superpixel's sampling probability")
    parser.add_argument("--sampling-num", type=int, default=250, help="Number of superpixel samples for linear model train")
    
    parser.add_argument("--quickshift-kernel", type=int, default=4, help="Quickshift kernel constant value")
    parser.add_argument("--quickshift-max-dist", type=int, default=80, help="Quickshift max distance value")
    parser.add_argument("--quickshift-ratio", type=float, default=0.2, help="Quickshift ratio value")
    
    parser.add_argument("--distance-kernel", type=float, default=0.25, help="Distance kernel constant value")
    
    parser.add_argument("--num-selected-coefs", type=int, default=5, help="Number of linear model coeficients used in LIME image")

    parser.add_argument("--save-image", action='store_true', help="Save image")
    
    args = parser.parse_args()

    lime_image = lime(
        args.path, 
        args.seed, 
        args.sampling_prob, 
        args.sampling_num,
        args.quickshift_kernel,
        args.quickshift_max_dist,
        args.quickshift_ratio,
        args.distance_kernel,
        args.num_selected_coefs
    )

    Image.fromarray(lime_image).show()

    if args.save_image:
        imsave("lime.png", lime_image)

if __name__ == "__main__":
    main()