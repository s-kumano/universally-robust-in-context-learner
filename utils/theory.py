from typing import Literal, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor

from .classifier import Transformer
from .data import (CIFAR10, FMNIST, MNIST, extract_all_data,
                   select_data_and_labels_binary)
from .utils import all_elements_in_targets


def calc_adv_threshold(d: int, lam: float) -> float:
    return ( 1 + (d - 1) * (lam / 2) ) / d


def calc_str_adv_threshold(d: int, lam: float) -> float:
    return (lam / 2) + (3/2) * (2 - lam) / ( (d - 1) * lam**2 + 3 )


def check_demos_queries(transformer: Transformer, demos: Tensor, queries: Tensor) -> None:
    assert len(demos.shape) == 3, demos.shape # (b, d+1, N_demos)
    assert len(queries.shape) == 3, queries.shape # (b, d, N_queries) or (b, d+1, N_queries)
    assert 0 not in demos.shape, demos.shape
    assert 0 not in queries.shape, queries.shape
    assert all_elements_in_targets(demos[:, -1], [-1, 1]), demos[:, -1].unique() # are all demo labels -1 or 1?
    if transformer.efficient:
        assert demos.shape[1] - 1 == queries.shape[1], (demos.shape, queries.shape)
    else:
        assert queries[:, -1].bool().logical_not().all(), queries[:, -1].unique() # are all query labels zero?
        assert demos.shape[1] == queries.shape[1], (demos.shape, queries.shape)


def check_gts(gts: Tensor) -> None:
    assert len(gts.shape) == 2, gts.shape # (b, N_queries)
    assert 0 not in gts.shape, gts.shape
    assert all_elements_in_targets(gts, [-1, 1]), gts.unique()


def predict(transformer: Transformer, demos: Tensor, queries: Tensor) -> Tensor:
    check_demos_queries(transformer, demos, queries)
    return transformer(demos, queries)[:, -1]


@torch.no_grad()
def attack(transformer: Transformer, demos: Tensor, queries: Tensor, gts: Tensor, eps: float) -> Tensor:
    check_demos_queries(transformer, demos, queries)
    check_gts(gts)

    Q = transformer.Q if transformer.efficient else transformer.Q[:, :-1]
    grads = transformer.P[-1:] @ demos @ demos.transpose(1, 2) @ Q # (b, 1, d)
    grads = grads[:, 0] # (b, d)
    batch_grads = - gts[:, None, :] * grads[..., None] # (b, d, N_queries)

    if not transformer.efficient:
        batch_grads = F.pad(batch_grads, (0, 0, 0, 1))

    return queries + eps * batch_grads.sign()


def generate_demos_queries_gts_training(
    batch_size: int, 
    d: int, 
    N_demos: int, 
    N_queries: int, 
    lam: float, 
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    assert lam > 0, lam

    c = torch.randint(0, d, (batch_size,), device=device) # (b,)

    N = N_demos + N_queries

    Z = torch.empty(batch_size, d+1, N, device=device) # (b, d+1, N)
    Z[:, :d] = torch.rand(batch_size, d, N) * lam
    Z[range(batch_size), c] = 1
    Z[:, -1] = 2 * torch.randint(0, 2, (batch_size, N), device=device) - 1
    Z[:, :d] *= Z[:, None, -1]

    demos = Z[..., :N_demos] # (b, d+1, N_demos)
    queries = Z[..., N_demos:] # (b, d+1, N_queries)
    gts = queries[:, -1].clone() # (b, N_queries)
    queries[:, -1] = 0

    return demos, queries, gts


def generate_demos_queries_gts_normal(
    batch_size: int, 
    d_rob: int, 
    d_vul: int,
    d_irr: int,
    N_demos: int, 
    N_queries: int, 
    alpha: float,
    beta: float,
    gamma: float,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    assert d_rob >= 1, d_rob
    assert d_vul >= 0, d_vul
    assert d_irr >= 0, d_irr
    assert alpha > 0, alpha
    assert beta >= 0, beta
    assert gamma >= 0, gamma

    d = d_rob + d_vul + d_irr
    N = N_demos + N_queries
    Z = torch.empty(batch_size, d+1, N, dtype=torch.float, device=device) # (b, d+1, N)

    randperm = torch.stack([torch.randperm(d) for _ in range(batch_size)]) # (b, d)

    d_rob_indices = randperm[:, :d_rob] # (b, d_rob)
    d_vul_indices = randperm[:, d_rob:d_rob+d_vul] # (b, d_vul)
    d_irr_indices = randperm[:, d_rob+d_vul:d_rob+d_vul+d_irr] # (b, d_irr)

    indices_rob = torch.arange(batch_size).view(batch_size, 1).expand(batch_size, d_rob)
    indices_vul = torch.arange(batch_size).view(batch_size, 1).expand(batch_size, d_vul)
    indices_irr = torch.arange(batch_size).view(batch_size, 1).expand(batch_size, d_irr)

    Z[indices_rob, d_rob_indices] = torch.normal(mean=alpha, std=alpha, size=(batch_size, d_rob, N), device=device)
    Z[indices_vul, d_vul_indices] = torch.normal(mean=beta, std=beta, size=(batch_size, d_vul, N), device=device)
    Z[indices_irr, d_irr_indices] = torch.normal(mean=0, std=gamma, size=(batch_size, d_irr, N), device=device)

    Z[:, -1] = 2 * torch.randint(0, 2, (batch_size, N), device=device) - 1
    Z[:, :d] *= Z[:, None, -1]

    demos = Z[..., :N_demos] # (b, d+1, N_demos)
    queries = Z[..., N_demos:] # (b, d+1, N_queries)
    gts = queries[:, -1].clone() # (b, N_queries)
    queries[:, -1] = 0

    return demos, queries, gts

    
def generate_demos_queries_gts_real(
    dataset_name: Literal['MNIST', 'FMNIST', 'CIFAR10'], 
    dataset_root: str, 
) -> Tuple[Tensor, Tensor, Tensor]:
    
    if dataset_name == 'MNIST':
        dataset_cls = MNIST
        kwargs = {}
    elif dataset_name == 'FMNIST':
        dataset_cls = FMNIST
        kwargs = {}
    elif dataset_name == 'CIFAR10':
        dataset_cls = CIFAR10
        kwargs = {'transform': T.ToTensor()}

    train_dataset = dataset_cls(dataset_root, True, **kwargs)
    test_dataset = dataset_cls(dataset_root, False, **kwargs)

    binary_labels = [(i, j) for i in range(10) for j in range(i+1, 10)]

    train_data, train_labels = extract_all_data(train_dataset, False)
    test_data, test_labels = extract_all_data(test_dataset, False)

    demos_list = []
    queries_list = []
    gts_list = []

    for binary_label in binary_labels:

        selected_train_data, selected_train_labels = select_data_and_labels_binary(train_data, train_labels, *binary_label)
        selected_test_data, selected_test_labels = select_data_and_labels_binary(test_data, test_labels, *binary_label)

        selected_train_data = selected_train_data.flatten(1) # (N_demos, d)
        selected_test_data = selected_test_data.flatten(1) # (N_queries, d)

        mean = selected_train_data.mean(0)
        selected_train_data -= mean
        selected_test_data -= mean

        sign = (selected_train_data * selected_train_labels[:, None]).sum(0).sign()
        selected_train_data *= sign
        selected_test_data *= sign

        demos = torch.concat([selected_train_data, selected_train_labels[:, None]], dim=1) # (N_demos, d+1)
        queries = torch.concat([selected_test_data, torch.zeros(selected_test_data.shape[0], 1)], dim=1) # (N_queries, d+1)

        demos_list.append(demos)
        queries_list.append(queries)
        gts_list.append(selected_test_labels)

    min_N_demos = min([demos.shape[0] for demos in demos_list])
    demos_list = [demos[:min_N_demos] for demos in demos_list]

    min_N_queries = min([queries.shape[0] for queries in queries_list])
    queries_list = [queries[:min_N_queries] for queries in queries_list]

    gts_list = [gts[:min_N_queries] for gts in gts_list]

    demos = torch.stack(demos_list) # (45, N_demos, d+1)
    queries = torch.stack(queries_list) # (45, N_queries, d)
    gts= torch.stack(gts_list) # (45, N_queries)

    demos = demos.transpose(1, 2)
    queries = queries.transpose(1, 2)

    return demos, queries, gts