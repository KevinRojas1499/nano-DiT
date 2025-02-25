import os
import numpy as np
import torch
import click
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import FolderOrZipDataset
from diffusers.models import AutoencoderKL
import distributed 
import json

@click.command()
@click.option('--path', type=str, default='dataset/ILSVRC/Data/CLS-LOC/train/')
@click.option('--dest', type=str, default='dataset/latent-imagenet')
@click.option('--size', type=int, default=256)
@click.option('--batch_size', type=int, default=16)
@click.option('--num_workers', type=int, default=4)
@click.option('--vae_opt', type=click.Choice(['ema', 'mse']), default='ema')
def process(path, dest, size, batch_size, num_workers, vae_opt):
    os.makedirs(dest, exist_ok=True)
    device, rank, world_size, seed = distributed.initialize_dist(0)
    
    
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_opt}").to(device)
    dataset = FolderOrZipDataset(path, size=size)
    sampler = DistributedSampler(dataset, world_size,rank, shuffle=False, seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, num_workers=num_workers)
    
    data_index = {}
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs, cond, idx = batch
            imgs = imgs.to(device)
            flipped_imgs = torch.flip(imgs, dims=[-1])

            dist = vae.encode(imgs)['latent_dist']
            latents = torch.cat((dist.mean, dist.std), dim=1)

            dist_flipped = vae.encode(flipped_imgs)['latent_dist']
            flipped_latents = torch.cat((dist_flipped.mean, dist_flipped.std), dim=1)

            for latent,flipped_latent, cond, i, in  zip(latents, flipped_latents, cond, idx):
                i_str = f'{i :08d}'
                folder = os.path.join(dest, f'{i_str[:4]}')
                os.makedirs(folder, exist_ok=True)
                
                output_path = os.path.join(folder, f'{i_str}.npy')
                np.save(output_path, latent.cpu().detach().numpy())
                flip_output_path = os.path.join(folder, f'flip_{i_str}.npy')
                np.save(flip_output_path, flipped_latent.cpu().detach().numpy())
                data_index[i.item()] = {
                    'img' : [output_path, flip_output_path],
                    'cond' : cond.item()
                }
            

        # Gather all data indexes from different ranks
        all_data_indexes = [None] * world_size
        distributed.gather_object(all_data_indexes, data_index) # Merge in rank 0 
        
        if rank == 0:
            merged_data_index = {}
            for rank_data in all_data_indexes:
                merged_data_index.update(rank_data)
            
            index_path = os.path.join(dest, 'data_index.json')
            with open(index_path, 'w') as f:
                json.dump(merged_data_index, f, indent=4)
            
        distributed.destroy()

if __name__ == "__main__":
    process()

