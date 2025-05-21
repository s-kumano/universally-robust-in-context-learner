import argparse
import os

import torch
from pytorch_lightning.lite import LightningLite
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils.classifier import Transformer
from utils.theory import attack, generate_demos_queries_gts_training, predict
from utils.utils import (print_used_gpu_memory, save_dict_as_files, set_seed,
                         to_cpu)


class Main(LightningLite):
    def run(
        self,
        batch_size: int,
        d: int,
        N: int,
        lam: float,
        eps: float,
        epochs: int,
        lr: float,
        seed: int,
    ) -> None:
        
        dir_name = f'batch_size={batch_size}_d={d}_N={N}_lam={lam}_eps={eps}_epochs={epochs}_lr={lr}_seed={seed}'
        dir_path = os.path.join('data', 'train', dir_name)

        if os.path.exists(dir_path):
            print(f'already exist: {dir_path}')
            return
        else:
            os.makedirs(dir_path, exist_ok=True)
        
        set_seed(seed)

        transformer = Transformer(d)
        optimizer = SGD(transformer.parameters(), lr, .9)
        scheduler = ReduceLROnPlateau(optimizer)

        transformer, optimizer = self.setup(transformer, optimizer)

        for _ in tqdm(range(epochs), mininterval=60):
            optimizer.zero_grad(True)

            demos, queries, gts = generate_demos_queries_gts_training(batch_size, d, N, 1, lam, self.device)

            queries = attack(transformer, demos, queries, gts, eps) if eps > 0 else queries

            loss = - (gts * predict(transformer, demos, queries)).mean()

            self.backward(loss)
            optimizer.step()
            scheduler.step(loss)

            with torch.no_grad():
                for param in transformer.parameters():
                    param.clamp_(0, 1)

        save_data = {'params': transformer}
        to_cpu(save_data)
        save_dict_as_files(save_data, dir_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int)
    parser.add_argument('d', type=int)
    parser.add_argument('N', type=int)
    parser.add_argument('lam', type=float)
    parser.add_argument('eps', type=float)
    parser.add_argument('epochs', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('seed', type=int)
    parser.add_argument('device', type=int)
    args = parser.parse_args()

    lite_kwargs = {
        'accelerator': 'gpu',
        'devices': [args.device],
        'precision': 16,
    }

    del args.device # type: ignore

    Main(**lite_kwargs).run(**vars(args))

    print_used_gpu_memory()