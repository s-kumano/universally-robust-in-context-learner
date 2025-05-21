from typing import Any, Callable, List, Optional, Tuple

import torch
import torchvision
import torchvision.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


def dataloader(
    dataset: Dataset, 
    batch_size: int, 
    shuffle: bool, 
    num_workers: int = 3, 
    pin_memory: bool = True, 
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def extract_all_data(dataset: Dataset, pin_memory: bool = True) -> Any:
    return next(iter(dataloader(dataset, len(dataset), False, 3, pin_memory, False))) # type: ignore


def select_data_and_labels(data: Tensor, labels: Tensor, target_labels: List[int]) -> Tuple[Tensor, Tensor]:
    assert len(data) == len(labels), (len(data), len(labels))
    assert len(target_labels) == len(set(target_labels)), target_labels

    indices = torch.isin(labels, torch.tensor(target_labels))

    data = data[indices]
    labels = labels[indices]

    target_labels = sorted(target_labels) # not to inplace
    for i, target_label in enumerate(target_labels):
        labels[labels == target_label] = i

    return data, labels


def select_data_and_labels_binary(
    data: Tensor, 
    labels: Tensor, 
    target_label_1: int, 
    target_label_2: int,
) -> Tuple[Tensor, Tensor]:
    data, labels = select_data_and_labels(data, labels, [target_label_1, target_label_2])
    labels[labels == 1] = -1
    labels[labels == 0] = 1
    return data, labels


class SequenceDataset(Dataset):
    def __init__(
        self, 
        xs: Any, 
        ys: Any, 
        x_transform: Optional[Callable] = None, 
        y_transform: Optional[Callable] = None,
    ) -> None:
        assert len(xs) == len(ys)
        self.xs, self.ys = xs, ys
        self.x_transform, self.y_transform = x_transform, y_transform

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x = self.x_transform(self.xs[idx]) if self.x_transform else self.xs[idx]
        y = self.y_transform(self.ys[idx]) if self.y_transform else self.ys[idx]
        return x, y
    

# `mean` and `std` must be list (not tuple)
# to be compatible with the type hint of `torchvision.transforms.functional.normalize`.
    

class MNIST(torchvision.datasets.MNIST):
    name = 'MNIST'
    mean = [.1307]
    std = [.3081]
    n_classes = 10
    classes = tuple(range(10))
    size = (1, 28, 28)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        transform = transform if transform else T.ToTensor()
        super().__init__(root, train, transform, target_transform, True)


class FMNIST(torchvision.datasets.FashionMNIST):
    name = 'FMNIST'
    mean = [.2860]
    std = [.3530]
    n_classes = 10
    #classes = ('T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    classes = ('Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot')
    size = (1, 28, 28)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        transform = transform if transform else T.ToTensor()
        super().__init__(root, train, transform, target_transform, True)


class CIFAR10(torchvision.datasets.CIFAR10):
    name = 'CIFAR10'
    mean = [.4914, .4822, .4465]
    std = [.2470, .2435, .2616]
    n_classes = 10
    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    size = (3, 32, 32)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        
        if transform is None:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ]) if train else T.ToTensor()

        super().__init__(root, train, transform, target_transform, True)