import os
import random
from typing import Any, Dict, List, Literal, Union

import numpy as np
import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchvision.transforms.functional import normalize
from tqdm import tqdm


def set_seed(seed: int = 0) -> None:
    #from lightning_lite.utilities.seed import seed_everything
    #seed_everything(seed, True)
    os.environ['PL_GLOBAL_SEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PL_SEED_WORKERS'] = str(1)


def gpu(id: int) -> torch.device:
    print(torch.cuda.get_device_name(id))
    return torch.device(f'cuda:{id}')


def print_used_gpu_memory():
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f'Allocated: {allocated:.2f} MB')
    print(f'Reserved: {reserved:.2f} MB')


def to_cpu(d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = v.cpu()
        elif isinstance(v, torch.nn.Module):
            d[k] = v.cpu().state_dict()
    return d


def save_dict_as_files(dict: Dict[str, Any], root: str) -> None:
    os.makedirs(root, exist_ok=True)
    for k, v in dict.items():
        p = os.path.join(root, k)
        torch.save(v, p)


@torch.no_grad()
def in_range(x: Tensor, min: float, max: float) -> bool:
    return ((min<=x.min()) & (x.max()<=max)).item() # type: ignore


@torch.no_grad()
def at_least_one_element_in_targets(x: Tensor, targets: List[float]) -> bool:
    return torch.isin(x, torch.tensor(targets, device=x.device)).any().item() # type: ignore


@torch.no_grad()
def all_elements_in_targets(x: Tensor, targets: List[float]) -> bool:
    return torch.isin(x, torch.tensor(targets, device=x.device)).all().item() # type: ignore


def freeze(model: Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze(model: Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def get_model_device(model: Module) -> torch.device:
    return next(model.parameters()).device


class ModelWithNormalization(Module):
    def __init__(self, model: Module, mean: List[float], std: List[float]) -> None:
        super().__init__()
        self.model = model
        self.mean, self.std = mean, std

    def forward(self, x: Tensor) -> Tensor:
        assert in_range(x, 0, 1)
        return self.model(normalize(x, self.mean, self.std))


class CalcClassificationAcc(LightningLite):
    def run(
        self, 
        classifier: Module, 
        dataloader: DataLoader, 
        n_classes: int, 
        top_k: int = 1,
        average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro',
    ) -> Union[float, List[float]]:

        classifier = self.setup(classifier)
        dataloader = self.setup_dataloaders(dataloader) # type: ignore

        freeze(classifier)
        classifier.eval()

        metric = MulticlassAccuracy(n_classes, top_k, average)
        self.to_device(metric)

        for xs, labels in tqdm(dataloader):
            outs = classifier(xs)
            metric(outs, labels)
        
        acc = metric.compute()
        return acc.tolist() if average == 'none' else acc.item()