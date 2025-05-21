import argparse
import os
from typing import Literal, Optional

import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor

from utils.classifier import Transformer
from utils.theory import (attack, generate_demos_queries_gts_normal,
                          generate_demos_queries_gts_real,
                          generate_demos_queries_gts_training, predict)
from utils.utils import (freeze, print_used_gpu_memory, save_dict_as_files,
                         set_seed, to_cpu)


def _check_and_set_dpath(weight_name: str, dname: str) -> str:
    
    dpath = os.path.join('data', 'test', weight_name, dname)

    if os.path.exists(dpath):
        print(f'already exist: {dpath}')

        import sys
        sys.exit(0)

    else:
        os.makedirs(dpath, exist_ok=True)
        return dpath
    

def _setup_transformer(lit: LightningLite, weight_name: str, d: int) -> Transformer:

    transformer = Transformer(d, True)
    freeze(transformer)

    if weight_name in ('ideal_std', 'ideal_adv'):
        
        P = torch.ones(d+1)

        if weight_name == 'ideal_std':
            Q = torch.ones(d+1, d)

        elif weight_name == 'ideal_adv':
            Q = torch.zeros(d+1, d)
            Q[:-1] = torch.eye(d)

        transformer.P.copy_(P)
        transformer.Q.copy_(Q)

    else:
        weight_path = os.path.join('data', 'train', weight_name, 'params')
        params = torch.load(weight_path, map_location=lit.device)
        transformer.load_state_dict(params)

    transformer.P.copy_(transformer.P / d)
    transformer.Q.copy_(transformer.Q / d)
        
    return lit.setup(transformer)


def _calc_accs(
    transformer: Transformer,
    demos: Tensor,
    queries: Tensor,
    gts: Tensor,
    eps: float,
    dpath: str,
) -> None:
    
    queries = queries[:, :-1]
    queries = attack(transformer, demos, queries, gts, eps) if eps > 0 else queries

    accs = (gts == predict(transformer, demos, queries).sign()).float().mean(1)

    save_data = {'accs': accs}
    to_cpu(save_data)
    save_dict_as_files(save_data, dpath)


class Train(LightningLite):
    def run(
        self,
        weight_name: str,
        batch_size: int,
        d: int,
        N_demos: int,
        N_queries: int,
        lam: float,
        eps: float,
        seed: int,
    ) -> None:
        
        dname = f'batch_size={batch_size}_d={d}_N_demos={N_demos}_N_queries={N_queries}_lam={lam}_eps={eps}_seed={seed}'
        dpath = _check_and_set_dpath(weight_name, dname)

        set_seed(seed)

        transformer = _setup_transformer(self, weight_name, d)

        demos, queries, gts = generate_demos_queries_gts_training(
            batch_size,
            d,
            N_demos,
            N_queries,
            lam,
            self.device,
        )

        _calc_accs(transformer, demos, queries, gts, eps, dpath)


class Normal(LightningLite):
    def run(
        self,
        weight_name: str,
        batch_size: int,
        d_rob: int,
        d_vul: int,
        d_irr: int,
        N_demos: int,
        N_queries: int,
        alpha: float,
        beta: float,
        gamma: float,
        eps: float,
        seed: int,
    ) -> None:
        
        dname = f'batch_size={batch_size}_d_rob={d_rob}_d_vul={d_vul}_d_irr={d_irr}_N_demos={N_demos}_N_queries={N_queries}_alpha={alpha}_beta={beta}_gamma={gamma}_eps={eps}_seed={seed}'
        dpath = _check_and_set_dpath(weight_name, dname)

        set_seed(seed)

        d = d_rob + d_vul + d_irr

        transformer = _setup_transformer(self, weight_name, d)

        demos, queries, gts = generate_demos_queries_gts_normal(
            batch_size,
            d_rob,
            d_vul,
            d_irr,
            N_demos,
            N_queries,
            alpha,
            beta,
            gamma,
            self.device,
        )

        _calc_accs(transformer, demos, queries, gts, eps, dpath)


class Real(LightningLite):
    def run(
        self,
        weight_name: str,
        dataset_name: Literal['MNIST', 'FMNIST', 'CIFAR10'],
        eps: float,
        seed: int,
        N_demos: Optional[int] = None,
    ) -> None:
        assert N_demos is None or N_demos > 0
        
        dname = f'dataset_name={dataset_name}_N_demos={N_demos}_eps={eps}_seed={seed}'
        dpath = _check_and_set_dpath(weight_name, dname)

        set_seed(seed)

        if dataset_name == 'MNIST':
            d = 28 * 28
        elif dataset_name == 'FMNIST':
            d = 28 * 28
        elif dataset_name == 'CIFAR10':
            d = 3 * 32 * 32

        transformer = _setup_transformer(self, weight_name, d)

        dataset_root = os.path.join('..', 'datasets')
        demos, queries, gts = generate_demos_queries_gts_real(dataset_name, dataset_root)

        assert N_demos is None or N_demos <= demos.shape[2]
        demos = demos[..., :N_demos]

        demos, queries, gts = self.to_device((demos, queries, gts))

        _calc_accs(transformer, demos, queries, gts, eps, dpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_name')
    parser.add_argument('eps', type=float)
    parser.add_argument('seed', type=int)
    parser.add_argument('device', type=int)

    subparsers_data = parser.add_subparsers(dest='data', required=True)

    parser_train = subparsers_data.add_parser('train')
    parser_train.add_argument('batch_size', type=int)
    parser_train.add_argument('d', type=int)
    parser_train.add_argument('N_demos', type=int)
    parser_train.add_argument('N_queries', type=int)
    parser_train.add_argument('lam', type=float)

    parser_normal = subparsers_data.add_parser('normal')
    parser_normal.add_argument('batch_size', type=int)
    parser_normal.add_argument('d_rob', type=int)
    parser_normal.add_argument('d_vul', type=int)
    parser_normal.add_argument('d_irr', type=int)
    parser_normal.add_argument('N_demos', type=int)
    parser_normal.add_argument('N_queries', type=int)
    parser_normal.add_argument('alpha', type=float)
    parser_normal.add_argument('beta', type=float)
    parser_normal.add_argument('gamma', type=float)

    parser_real = subparsers_data.add_parser('real')
    parser_real.add_argument('dataset_name', choices=('MNIST', 'FMNIST', 'CIFAR10'))
    parser_real.add_argument('--N_demos', type=int)

    args = parser.parse_args()

    lite_kwargs = {
        'accelerator': 'gpu',
        'devices': [args.device],
        'precision': 16,
    }

    del args.device # type: ignore

    if args.data == 'train':
        del args.data # type: ignore
        Train(**lite_kwargs).run(**vars(args))

    elif args.data == 'normal':
        del args.data # type: ignore
        Normal(**lite_kwargs).run(**vars(args))

    elif args.data == 'real':
        del args.data # type: ignore
        Real(**lite_kwargs).run(**vars(args))

    print_used_gpu_memory()