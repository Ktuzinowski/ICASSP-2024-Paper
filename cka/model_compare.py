import os
import torch
import torchvision.models as models
from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2, swin_b
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from functools import partial
from warnings import warn
from typing import List, Dict
import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
import matplotlib.pyplot as plt

def add_colorbar(im, aspect=10, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot."""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

class CudaCKA(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

class CKA:
    def __init__(self,
                 model1: nn.Module,
                 model2: nn.Module,
                 model1_name: str = None,
                 model2_name: str = None,
                 model1_layers: List[str] = None,
                 model2_layers: List[str] = None,
                 device: str ='cpu'):
        """

        :param model1: (nn.Module) Neural Network 1
        :param model2: (nn.Module) Neural Network 2
        :param model1_name: (str) Name of model 1
        :param model2_name: (str) Name of model 2
        :param model1_layers: (List) List of layers to extract features from
        :param model2_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model1 = model1
        self.model2 = model2

        self.device = device
        self.cuda_cka = CudaCKA(device)

        self.model1_info = {}
        self.model2_info = {}

        if model1_name is None:
            self.model1_info['Name'] = model1.__repr__().split('(')[0]
        else:
            self.model1_info['Name'] = model1_name

        if model2_name is None:
            self.model2_info['Name'] = model2.__repr__().split('(')[0]
        else:
            self.model2_info['Name'] = model2_name

        if self.model1_info['Name'] == self.model2_info['Name']:
            warn(f"Both model have identical names - {self.model2_info['Name']}. " \
                 "It may cause confusion when interpreting the results. " \
                 "Consider giving unique names to the models :)")

        self.model1_info['Layers'] = []
        self.model2_info['Layers'] = []

        self.model1_features = {}
        self.model2_features = {}

        if len(list(model1.modules())) > 150 and model1_layers is None:
            warn("Model 1 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model1_layers' parameter. Your CPU/GPU will thank you :)")

        self.model1_layers = model1_layers

        if len(list(model2.modules())) > 150 and model2_layers is None:
            warn("Model 2 seems to have a lot of layers. " \
                 "Consider giving a list of layers whose features you are concerned with " \
                 "through the 'model2_layers' parameter. Your CPU/GPU will thank you :)")

        self.model2_layers = model2_layers

        self._insert_hooks()
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self.model1.eval()
        self.model2.eval()

    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model1":
            self.model1_features[name] = out

        elif model == "model2":
            self.model2_features[name] = out

        else:
            raise RuntimeError("Unknown model name for _log_layer.")

    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model1.named_modules():
            if self.model1_layers is not None:
                if name in self.model1_layers:
                    self.model1_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model1", name))
            else:
                self.model1_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model1", name))

        # Model 2
        for name, layer in self.model2.named_modules():
            if self.model2_layers is not None:
                if name in self.model2_layers:
                    self.model2_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model2", name))
            else:

                self.model2_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model2", name))

    def _HSIC(self, K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = torch.ones(N, 1).to(self.device)
        result = torch.trace(K @ L)
        result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
        result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
        return (1 / (N * (N - 3)) * result).item()

    # before layer normalizations
    # cls token comparisons
    # each attention for block compare output for them
    def compare_linear_CKA(self,
                           dataloader1: DataLoader,
                           dataloader2: DataLoader = None) -> None:
        """
        Computes the linear CKA feature similarity between models on the given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """
        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1  

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))

        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        # Only have one channel since this is linear CKA
        self.hsic_matrix = torch.zeros(N, M)

        num_batches = min(len(dataloader1), len(dataloader1))

        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            maximum_size = 5000
            
            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                if len(feat1) == 2 and "self_attention" in name1:
                    feat1 = feat1[0]
                X = feat1.flatten(1)
                # print("Shape of X", X.shape)
                #X = X.transpose(0, 1)
                # if X.shape[0] > maximum_size:
                #     X = X[:5000, 0]
                # print("New Shape of X", X.shape)
    
                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    if len(feat2) == 2 and "self_attention" in name2:
                        feat2 = feat2[0]
                    Y = feat2.flatten(1)
                    # print("Shape of Y", Y.shape, Y)
                    #Y = Y.transpose(0, 1)
                    # if Y.shape[0] > maximum_size:
                    #     Y = Y[:5000, 0]
                    # print("New Shape of Y", Y.shape)

                    self.hsic_matrix[i, j] = self.cuda_cka.linear_CKA(X, Y)
                    print("CKA", self.hsic_matrix[i, j])

    def compare_token_pairwise_CKA(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        # 768 * number of layers
        number_of_comparisons_model1 = 768 * N
        number_of_comparisons_model2 = 768 * M
        self.hsic_matrix = torch.zeros(number_of_comparisons_model1, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))
        print("Am I working?")
        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
            print("Idk are you?")
            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                print(f"Name of feature {name1}")
                if len(feat1) == 2 and "self_attention" in name1:
                    feat1 = feat1[0]
                # X = feat1.flatten(1)
                # if X.shape[1] > X.shape[0]:
                #     print(f"Shape that is being changed {X.shape}")
                #     X = X.transpose(0, 1)
                print(f"Preprocessing shape of X={feat1.shape}")
                X = feat1[:, 0, :]
                print(f"Current Shape X={feat1.shape}")
                
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    print(f"Name of feature {name2}")
                    if len(feat2) == 2 and "self_attention" in name2:
                        feat2 = feat2[0]
                    Y = feat2[:, 0, :]
                    # Y = feat2.flatten(1)
                    
                    # if Y.shape[1] > Y.shape[0]:
                    #     print(f"Shape that is being changed {Y.shape}")
                    #     Y = Y.transpose(0, 1)
                    
                    print(f"Current Shape Y={Y.shape}")
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        self.hsic_matrix = torch.nan_to_num(self.hsic_matrix, nan=0.0)
        # assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"
        # Replace negative values with zero
        self.hsic_matrix = torch.clamp(self.hsic_matrix, min=0.0)

    def compare(self,
                dataloader1: DataLoader,
                dataloader2: DataLoader = None) -> None:
        """
        Computes the feature similarity between the models on the
        given datasets.
        :param dataloader1: (DataLoader)
        :param dataloader2: (DataLoader) If given, model 2 will run on this
                            dataset. (default = None)
        """

        if dataloader2 is None:
            warn("Dataloader for Model 2 is not given. Using the same dataloader for both models.")
            dataloader2 = dataloader1

        self.model1_info['Dataset'] = dataloader1.dataset.__repr__().split('\n')[0]
        self.model2_info['Dataset'] = dataloader2.dataset.__repr__().split('\n')[0]

        N = len(self.model1_layers) if self.model1_layers is not None else len(list(self.model1.modules()))
        M = len(self.model2_layers) if self.model2_layers is not None else len(list(self.model2.modules()))

        # 768 * number of layers
        number_of_comparisons_model1 = 768 * N
        number_of_comparisons_model2 = 768 * M
        self.hsic_matrix = torch.zeros(N, M, 3)

        num_batches = min(len(dataloader1), len(dataloader1))
        print("Am I working?")
        for (x1, *_), (x2, *_) in tqdm(zip(dataloader1, dataloader2), desc="| Comparing features |", total=num_batches):
            print("Idk are you?")
            self.model1_features = {}
            self.model2_features = {}
            _ = self.model1(x1.to(self.device))
            _ = self.model2(x2.to(self.device))

            for i, (name1, feat1) in enumerate(self.model1_features.items()):
                print(f"Name of feature {name1}")
                if len(feat1) == 2 and "self_attention" in name1:
                    feat1 = feat1[0]
                # X = feat1.flatten(1)
                # if X.shape[1] > X.shape[0]:
                #     print(f"Shape that is being changed {X.shape}")
                #     X = X.transpose(0, 1)
                print(f"Preprocessing shape of X={feat1.shape}")
                X = feat1[:, 0, :]
                print(f"Current Shape X={feat1.shape}")
                
                K = X @ X.t()
                K.fill_diagonal_(0.0)
                self.hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches

                for j, (name2, feat2) in enumerate(self.model2_features.items()):
                    print(f"Name of feature {name2}")
                    if len(feat2) == 2 and "self_attention" in name2:
                        feat2 = feat2[0]
                    Y = feat2[:, 0, :]
                    # Y = feat2.flatten(1)
                    
                    # if Y.shape[1] > Y.shape[0]:
                    #     print(f"Shape that is being changed {Y.shape}")
                    #     Y = Y.transpose(0, 1)
                    
                    print(f"Current Shape Y={Y.shape}")
                    L = Y @ Y.t()
                    L.fill_diagonal_(0)
                    assert K.shape == L.shape, f"Feature shape mistach! {K.shape}, {L.shape}"

                    self.hsic_matrix[i, j, 1] += self._HSIC(K, L) / num_batches
                    self.hsic_matrix[i, j, 2] += self._HSIC(L, L) / num_batches

        self.hsic_matrix = self.hsic_matrix[:, :, 1] / (self.hsic_matrix[:, :, 0].sqrt() *
                                                        self.hsic_matrix[:, :, 2].sqrt())

        self.hsic_matrix = torch.nan_to_num(self.hsic_matrix, nan=0.0)
        # assert not torch.isnan(self.hsic_matrix).any(), "HSIC computation resulted in NANs"
        # Replace negative values with zero
        self.hsic_matrix = torch.clamp(self.hsic_matrix, min=0.0)

    def export(self) -> Dict:
        """
        Exports the CKA data along with the respective model layer names.
        :return:
        """
        return {
            "model1_name": self.model1_info['Name'],
            "model2_name": self.model2_info['Name'],
            "CKA": self.hsic_matrix,
            "model1_layers": self.model1_info['Layers'],
            "model2_layers": self.model2_info['Layers'],
            "dataset1_name": self.model1_info['Dataset'],
            "dataset2_name": self.model2_info['Dataset'],

        }

    def plot_results(self,
                     save_path: str = None,
                     title: str = None):
        self.hsic_matrix = self.hsic_matrix.detach().numpy()
        print(self.hsic_matrix)
        fig, ax = plt.subplots()
        im = ax.imshow(self.hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        if title is not None:
            ax.set_title(f"{title}", fontsize=18)
        else:
            ax.set_title(f"{self.model1_info['Name']} vs {self.model2_info['Name']}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def get_accuracy_for_models(self, accuracy_dataloader):
        correct_model1 = 0
        total_model1 = 0

        correct_model2 = 0
        total_model2 = 0

        with torch.no_grad():  # Disable gradient calculation for inference
            for data in accuracy_dataloader:
                inputs, labels = data

                # Move inputs and labels to the device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Model 1 Predictions
                outputs_model1 = self.model1(inputs)
                _, predicted_model1 = torch.max(outputs_model1, 1)
                total_model1 += labels.size(0)
                correct_model1 += (predicted_model1 == labels).sum().item()

                # Model 2 Predictions
                outputs_model2 = self.model2(inputs)
                _, predicted_model2 = torch.max(outputs_model2, 1)
                total_model2 += labels.size(0)
                correct_model2 += (predicted_model2 == labels).sum().item()

        accuracy_model1 = 100 * correct_model1 / total_model1
        accuracy_model2 = 100 * correct_model2 / total_model2
        print(f"{self.model1_info['Name']} Accuracy of the network on the test images: {accuracy_model1} %")
        print(f"{self.model2_info['Name']} Accuracy of the network on the test images: {accuracy_model2} %")

        return (accuracy_model1, accuracy_model2)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
#===============================================================
batch_size = 100
arch = "vit_b_16"
pretrained = True

model1 = models.__dict__[arch]()
model2 = models.__dict__[arch]()

state_dict_model_two = torch.load('checkpoints/checkpoint_89.pth')

new_state_dict = {}
for key, value in state_dict_model_two.items():
    new_key = key.replace('module.', '')  # Remove 'module.' from the key
    new_state_dict[new_key] = value
model2.load_state_dict(new_state_dict)
model1.load_state_dict(new_state_dict)

path_to_imagenet = "/home/idies/workspace/Temporary/ktuzinows1/scratch/imagenet"
val_dir = os.path.join(path_to_imagenet, "val")
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_dataset = datasets.ImageFolder(
    val_dir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
]))
# Define the number of samples you want in the smaller dataset
# USE LARGE BATCH SIZE
num_samples = 10000

# Generate a random list of indices
indices = np.random.choice(len(val_dataset), num_samples, replace=False)

# Create the subset
small_val_dataset = Subset(val_dataset, indices)

val_sampler = None
val_loader = torch.utils.data.DataLoader(
        small_val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=val_sampler)

model1_layer_names = []
# CLS token here
# encoder.layers.encoder_layer_0.self_attention.out_proj
# counter = 2
# TODO: Compare tokens pairwise instead of just CLS token
# TODO: Possibly Average Pool 768 dimension embedding
# TODO (1): Print outputs of latent representations, and trick to reduce dimensions
# TODO: Use pretrained model on imagenet, plot mean attention distance for CIFAR10, deep layers high attention/small layers lower attention
# Include also from scratch ViT on CIFAR10
for name, layer in model1.named_modules():
    if 'mlp.4' in name:
        model1_layer_names.append(name)
    # if counter == 0:
    #     break
    # counter -= 1
print("Layer names for model1", model1_layer_names)
model_name = "ViT-B/16 0%"
model_name1 = "ViT-B/16 100%"
# torch.cuda.set_device(3)
cka = CKA(model1, model2,
        model1_name=model_name, model2_name=model_name1,
        device='cuda', model1_layers=model1_layer_names, model2_layers=model1_layer_names)
model1_accuracy, model2_accuracy = cka.get_accuracy_for_models(val_loader)
print("This is the model1 accuracy", model1_accuracy, "This is the model2 accuracy", model2_accuracy)
cka.compare(val_loader)
cka.plot_results(save_path="ViT_B_16_0_vs_100.png")
