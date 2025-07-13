import torch
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from tqdm import tqdm
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args


def update_ema(ema_model: Module, model: Module, decay: float = 0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model: Module, flag: bool = True):
    for p in model.parameters():
        p.requires_grad = flag


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    y0 = (arr.shape[0] - image_size) // 2
    x0 = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[y0:y0+image_size, x0:x0+image_size])


class RandomPairDataset(Dataset):
    """
    Generates random image pairs for testing the flow matching pipeline.
    """
    def __init__(self, length: int, image_size: int, transform=None):
        self.length = length
        self.image_size = image_size
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        arr1 = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
        arr2 = np.random.randint(0, 256, (self.image_size, self.image_size, 3), dtype=np.uint8)
        img1 = Image.fromarray(arr1)
        img2 = Image.fromarray(arr2)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2


def create_logger(log_dir: str):
    handlers = [logging.StreamHandler()]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        handlers.append(logging.FileHandler(os.path.join(log_dir, 'log.txt')))
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        handlers=handlers)
    return logging.getLogger(__name__)


def main(args):
    device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    logger = create_logger(args.results_dir)
    logger.info(f"Using device: {device}")

    # Experiment dir
    os.makedirs(args.results_dir, exist_ok=True)
    idx = len(glob(f"{args.results_dir}/*"))
    name = args.model.replace('/', '-')
    exp = f"{idx:03d}-{name}-{args.path_type}-{args.prediction}-{args.loss_weight}"
    exp_dir = os.path.join(args.results_dir, exp)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger.info(f"Experiment dir: {exp_dir}")

    # Model and EMA
    latent_size = args.image_size // 8
    model = SiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device)
    ema = deepcopy(model)
    requires_grad(ema, False)
    ema = ema.to(device)
    
    # Optional ckpt
    if args.ckpt and os.path.isfile(args.ckpt):
        state = find_model(args.ckpt)
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        logger.info(f"Loaded checkpoint from {args.ckpt}")

    # Transport and VAE
    transport = create_transport(
        args.path_type, args.prediction, args.loss_weight,
        args.train_eps, args.sample_eps)
    sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Random dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    dataset = RandomPairDataset(length=1000, image_size=args.image_size, transform=transform)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=args.num_workers,
                         pin_memory=True, drop_last=True)
    logger.info(f"Random dataset size: {len(dataset)} pairs")

    # Training loop: unconditional flow matching
    train_steps = 0
    start = time()
    logger.info(f"Training for {args.epochs} epochs (unconditional flow)...")
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch} start")
        for img1, img2 in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
            img1, img2 = img1.to(device), img2.to(device)
            with torch.no_grad():
                z0 = vae.encode(img1).latent_dist.sample().mul_(0.18215)
                z1 = vae.encode(img2).latent_dist.sample().mul_(0.18215)

            # Random time t
            B = z0.size(0)
            t = torch.rand(B, device=device)            # shape [B]
            # Interpolate
            t_broadcast = t.view(B, *([1]*(z0.dim()-1)))
            zt = (1 - t_broadcast) * z0 + t_broadcast * z1

            # Predict velocity: pass time and dummy label
            y = torch.zeros(B, dtype=torch.long, device=device)
            vt = model(zt, t, y)
            v_gt = (z1 - z0)
            loss = torch.nn.functional.mse_loss(vt, v_gt)

            opt.zero_grad(); loss.backward(); opt.step()
            update_ema(ema, model)

            train_steps += 1
            if train_steps % args.log_every == 0:
                elapsed = time() - start
                logger.info(f"Step {train_steps}, Loss: {loss.item():.4f}, Steps/sec: {args.log_every/elapsed:.2f}")
                start = time()

            if train_steps % args.ckpt_every == 0:
                ckpt = os.path.join(ckpt_dir, f"{train_steps:07d}.pt")
                torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'opt': opt.state_dict(), 'args': args}, ckpt)
                logger.info(f"Saved checkpoint: {ckpt}")

            if train_steps % args.sample_every == 0:
                logger.info("Sampling from z0 to z1...")
                sample_fn = sampler.sample_ode()
                y_sample = torch.zeros(z0.size(0), dtype=torch.long, device=device)
                traj = sample_fn(z0, ema.forward, y=y_sample)
                zT = traj[-1]
                imgs = vae.decode(zT / 0.18215).sample
                logger.info("Sampling done.")

    logger.info("Training complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--model', type=str, choices=list(SiT_models.keys()), default='SiT-S/2')
    parser.add_argument('--image-size', type=int, choices=[256,512], default=256)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--global-seed', type=int, default=0)
    parser.add_argument('--vae', choices=['ema','mse'], default='ema')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--ckpt-every', type=int, default=500)
    parser.add_argument('--sample-every', type=int, default=200)
    parser.add_argument('--cfg-scale', type=float, default=1.0)
    parser.add_argument('--ckpt', type=str, default=None)
    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
