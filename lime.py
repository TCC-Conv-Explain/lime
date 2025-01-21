import numpy as np

from skimage.segmentation import quickshift

import torch
from torch import argmax
from torch import Tensor
from torch.nn import Module

from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import ToTensor, ToPILImage

from sklearn.linear_model import LinearRegression

from PIL import Image

from tqdm import tqdm

import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Quickshifter:
    def __init__(
        self,
        kernel,
        max_dist,
        ratio
    ):
        self.kernel = kernel
        self.max_dist = max_dist
        self.ratio = ratio

        self.superpixels = None
        self.num_superpixels = None
    
    def compute(self, image: Tensor):
        if self.superpixels is not None:
            return
        
        image = image.permute(1, 2, 0).numpy()
        self.superpixels = torch.tensor(
            quickshift(
                image, 
                kernel_size=self.kernel, 
                max_dist=self.max_dist, 
                ratio=self.ratio
            )
        )
        self.num_superpixels = len(set(self.superpixels.flatten().tolist()))

class SuperpixelSampler:
    def __init__(
        self,
        quickshifter: Quickshifter,
        probability: float,
        num_samples: int,
        seed: int
    ):
        self.quickshifter = quickshifter
        self.probability = probability
        self.num_samples = num_samples
        self.seed = seed

        self.sample = None
        self.image_sample = None

    def compute(self, image: Tensor):
        if self.sample is not None:
            return
        
        self.quickshifter.compute(image)

        torch.manual_seed(self.seed)
        probability_tensor = torch.full((self.num_samples, self.quickshifter.num_superpixels), self.probability)
        self.sample = torch.bernoulli(probability_tensor).to(dtype=torch.int)
        self.image_sample = self._image_batch_from_sample(image)
    
    def _image_batch_from_sample(self, image):
        batch_size = self.sample.size(0)
        
        masks = []
        print("Building image batch ...")
        for i in tqdm(range(batch_size)):
            sample_indexes = torch.nonzero(self.sample[i] == 1, as_tuple=True)
            mask = torch.isin(self.quickshifter.superpixels, torch.cat(sample_indexes))
            masks.append(mask)
        
        masks = torch.stack(masks)  # Stack masks to match batch dimension

        return image.unsqueeze(0) * masks.unsqueeze(1).expand(-1, 3, -1, -1)

    def image_from_superpixels(self, image, superpixels_indices):
        superpixels = self.quickshifter.superpixels

        masked_superpixels = torch.isin(superpixels, torch.tensor(superpixels_indices)).float()
        return image * masked_superpixels

class SampleWeightsCalculator:
    def __init__(self, kernel):
        self.kernel = kernel
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

    def compute(self, superpixels):
        distances = self.cosine_similarity(torch.ones_like(superpixels).float(), superpixels)
        return torch.sqrt(torch.exp(- distances**2 / self.kernel**2))

class Lime:
    def __init__(
        self,
        superpixel_sampler: SuperpixelSampler,
        model: Module,
        sample_weights_calculator: SampleWeightsCalculator,
    ):
        self.model = model
        self.model.eval()
        self.superpixel_sampler = superpixel_sampler
        self.sample_weights_calculator = sample_weights_calculator
        self.linear_model = LinearRegression()
    
    @torch.no_grad()
    def _find_explained_class(self, image: Tensor):
        return argmax(self.model(image))

    @torch.no_grad()
    def _compute_explained_class(self, image: Tensor):
        self.model.eval()

        return torch.argmax(self.model(image.unsqueeze(0).to(DEVICE))[0]).item()

    @torch.no_grad()
    def _compute_model_preds(self, batch_image: Tensor, explained_class: int):
        assert len(batch_image.size()) == 4
        self.model.eval()
        
        return self.model(batch_image.to(DEVICE))[:, explained_class]

    def train(self, image):
        self.superpixel_sampler.compute(image)
        explained_class = self._compute_explained_class(image)
        print("Computing model predictions")
        preds = self._compute_model_preds(self.superpixel_sampler.image_sample, explained_class)
        sample_weights = self.sample_weights_calculator.compute(self.superpixel_sampler.sample)
        
        print("Training Linear model")
        self.linear_model.fit(
            X=self.superpixel_sampler.sample.cpu(), 
            y=preds.cpu(), 
            sample_weight=sample_weights
        )

        return self.linear_model.coef_


def main(args):
    image = Image.open(args.path)
    image = ToTensor()(image)

    quickshifter = Quickshifter(args.quickshift_kernel, args.quickshift_max_dist, args.quickshift_ratio)
    model = vgg16(weights=VGG16_Weights.DEFAULT).to(DEVICE)

    superpixel_sampler = SuperpixelSampler(quickshifter, args.sampling_prob, args.sampling_num, args.seed)
    sample_weights_calculator = SampleWeightsCalculator(args.distance_kernel)
    lime = Lime(superpixel_sampler, model, sample_weights_calculator)

    coefs = lime.train(image)
    top_features = np.argsort(coefs)[-args.num_selected_coefs:]

    ToPILImage()(superpixel_sampler.image_from_superpixels(image, top_features)).show()


if __name__ == "__main__":
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

    parser.add_argument("--save-image", type=str, help="Saved image path")
    
    args = parser.parse_args()
    main(args)