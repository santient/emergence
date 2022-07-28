from abc import ABC, abstractmethod
import argparse
import os
import shutil
import sys
import hashlib
import time
import math
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn.init import normal_, xavier_normal_
import torch.nn.functional as F
from PIL import Image
from PIL.ImageQt import ImageQt
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
import librosa
import subprocess
import warnings

class Effect(ABC):
    def __init__(self, args):
        self.buffer = None
    @abstractmethod
    def apply(self, world):
        pass

class Identity(Effect):
    def apply(self, world):
        return world

class Invert(Effect):
    def apply(self, world):
        return -world

class Sobel(Effect):
    def __init__(self, args):
        super().__init__(args)
        device = get_device(args.device)
        self.kx = torch.tensor([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]]).view(1, 1, 1, 3, 3).to(device)
        self.ky = torch.tensor([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]]).view(1, 1, 1, 3, 3).to(device)
    def apply(self, world):
        padded = F.pad((world + 1) / 2, (1, 1, 1, 1), mode="reflect")
        unfold = padded.unfold(1, 3, 1).unfold(2, 3, 1)
        gx = (unfold * self.kx).sum(dim=(3, 4))
        gy = (unfold * self.ky).sum(dim=(3, 4))
        mag = torch.sqrt(gx.square() + gy.square())
        return mag / mag.max() * 2 - 1

class Glow(Effect):
    def __init__(self, args):
        super().__init__(args)
        self.sobel = Sobel(args)
    def apply(self, world):
        return 0.5 * world + 0.5 * self.sobel.apply(world)

class Grayscale(Effect):
    def apply(self, world):
        return world.mean(dim=0, keepdim=True).expand(3, -1, -1)

class Saturate(Effect):
    def apply(self, world):
        sat = (world + 1) / 2
        maxi, _ = sat.max(dim=0, keepdim=True)
        mini, _ = sat.min(dim=0, keepdim=True)
        return (sat - mini) / (maxi - mini) * 2 - 1

class Desaturate(Effect):
    def apply(self, world):
        return world / 2

class Normalize(Effect):
    def apply(self, world):
        norm = (world + 1) / 2
        bot = torch.quantile(norm, 0.01)
        top = torch.quantile(norm, 0.99)
        return torch.clamp((norm - bot) / (top - bot), 0, 1) * 2 - 1

class SBlur(Effect):
    def apply(self, world):
        padded = F.pad(world, (1, 1, 1, 1), mode="reflect")
        unfold = padded.unfold(1, 3, 1).unfold(2, 3, 1)
        return unfold.mean(dim=(3, 4))

class MBlur(Effect):
    def apply(self, world):
        avg = world
        if self.buffer is not None:
            avg = (avg + self.buffer) / 2
        self.buffer = world
        return avg

class HMirror(Effect):
    def apply(self, world):
        return torch.cat((world[:, :, :world.size(2) // 2], world[:, :, :world.size(2) // 2].flip(2)), dim=2)

class VMirror(Effect):
    def apply(self, world):
        return torch.cat((world[:, :world.size(1) // 2, :], world[:, :world.size(1) // 2, :].flip(1)), dim=1)

effects_dict = {
    "identity": Identity,
    "invert": Invert,
    "sobel": Sobel,
    "glow": Glow,
    "grayscale": Grayscale,
    "saturate": Saturate,
    "desaturate": Desaturate,
    "normalize": Normalize,
    "sblur": SBlur,
    "mblur": MBlur,
    "hmirror": HMirror,
    "vmirror": VMirror,
}

def init_effects(effect_keys, args):
    effects = []
    if effect_keys is not None:
        for effect_key in effect_keys:
            effect = effects_dict[effect_key](args)
            effects.append(effect)
    return effects

def apply_effects(world, effects):
    fx = world
    for effect in effects:
        fx = effect.apply(fx)
    return fx

def get_filters(world, model):
    coords = 5 * torch.stack(torch.meshgrid(torch.linspace(-1, 1, world.size(1)), torch.linspace(-1, 1, world.size(2)), indexing="ij"), dim=-1).to(model.device)
    filters = model(coords)
    return filters

def create_model(out_dim, device):
    model = Sequential(
        Linear(2, out_dim),
        ReLU(),
        Linear(out_dim, out_dim)
        ).to(device)
    model.out_dim = out_dim
    model.device = device
    for p in model.parameters():
        if p.dim() > 1:
            xavier_normal_(p, 2)
        else:
            normal_(p, 0, 0.25)
    return model

def audio_features(pcm, args):
    hop = args.sr // args.fr
    h, p = librosa.effects.hpss(pcm)
    ons = librosa.onset.onset_strength(y=p, sr=args.sr, hop_length=hop)
    mel = librosa.feature.melspectrogram(y=h, sr=args.sr, n_mels=len(args.filter_sizes), hop_length=hop)
    db = librosa.power_to_db(mel, ref=np.max)
    features = np.concatenate((ons[np.newaxis, :], (db[::-1] + 80) / 80), axis=0)
    features = features.transpose(1, 0).copy()
    return torch.from_numpy(features)

def rgb_relu(world):
    mask = world.sum(dim=0, keepdim=True) > 0
    return world * mask

def step(world, model, delta, features, global_step, args):
    filters = get_filters(world, model)
    world_out = world
    start = 0
    for idx, size in enumerate(args.filter_sizes):
        if features is not None:
            scale = args.sensitivity * (features[global_step, 0] / 2 + features[global_step, idx + 1] * 2)
        else:
            scale = 1
        end = start + 3 * 3 * size * size + 12
        fil = scale * filters[:, :, start:end - 12].view(world.size(1), world.size(2), 3, 3, size, size)
        bias = filters[:, :, end - 12:end - 9].view(world.size(1), world.size(2), 3)
        lin = (scale / 2) * filters[:, :, end - 9:end].view(world.size(1), world.size(2), 3, 3)
        start = end
        pad = size // 2
        if pad > 0:
            padded = F.pad(world_out, (pad, pad, pad, pad), mode="reflect")
        else:
            padded = world_out
        unfold = padded.unfold(1, size, 1).unfold(2, size, 1)
        world_out = (fil.permute(2, 0, 1, 3, 4, 5) * unfold.unsqueeze(3)).sum(dim=(3, 4, 5))
        world_out = world_out + bias.permute(2, 0, 1)
        world_out = torch.relu(world_out)
        world_out = world_out + (lin.permute(2, 0, 1, 3) * world_out.unsqueeze(3)).sum(dim=3)
        if idx < len(args.filter_sizes) - 1:
            world_out = rgb_relu(world_out)
    world_out = torch.tanh(world_out)
    if args.interval > 0 and global_step % args.interval == 0:
        delta.clear()
        waypoint = create_model(model.out_dim, model.device)
        for p, w in zip(model.parameters(), waypoint.parameters()):
            delta.append((w.data - p.data) / args.interval)
    if delta:
        for p, d in zip(model.parameters(), delta):
            p.data = p.data + d
    return world_out

def to_img(world):
    img = (world + 1) / 2
    img = img * 255
    img = img.permute(1, 2, 0).cpu().numpy()
    img = np.round(img).astype("uint8")
    return Image.fromarray(img)

def abort():
    print("Aborted.")
    sys.exit(0)

def finish():
    print("Done!")
    sys.exit(0)

def get_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(name)
    return device

def close_preview(event):
    global preview_open
    preview_open = False

class ArgumentParserWithDefaults(argparse.ArgumentParser):
    def add_argument(self, *args, help=None, default=None, **kwargs):
        if help is not None:
            kwargs["help"] = help
        if default is not None and args[0] != "-h":
            kwargs["default"] = default
            if help is not None:
                kwargs["help"] += f" [default: {default}]"
        super().add_argument(*args, **kwargs)

def get_args():
    parser = ArgumentParserWithDefaults(
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=30))
    parser.add_argument("--audio_file", type=str, metavar="F",
        help="input audio file path")
    parser.add_argument("--out_file", type=str, required=True, metavar="F",
        help="output video file path")
    parser.add_argument("--out_dir", type=str, default="./out", metavar="D",
        help="output png directory for video frames")
    parser.add_argument("--seed_str", type=str, metavar="S",
        help="seed string (hashed into 64-bit integer)")
    parser.add_argument("--seed_int", type=int, metavar="I",
        help="64-bit seed integer")
    parser.add_argument("--video_dims", type=int, nargs=2, default=(720, 720), metavar=("W", "H"),
        help="output video dimensions")
    parser.add_argument("--video_length", type=int, metavar="L",
        help="manually specify video length in frames")
    parser.add_argument("--filter_sizes", type=int, nargs="+", default=(3, 5, 7), metavar="S",
        help="convolutional filter sizes")
    parser.add_argument("--fr", type=int, default=30,
        help="video frame rate")
    parser.add_argument("--sr", type=int, default=22050,
        help="audio sample rate")
    parser.add_argument("--sensitivity", type=float, default=1.0, metavar="S",
        help="audio reactive sensitivity")
    parser.add_argument("--interval", type=int, default=1800, metavar="I",
        help="evolution interval for model weights (0 for no evolution)")
    parser.add_argument("--effects", type=str, nargs="*", metavar="E",
        help=effects_help)
    parser.add_argument("--device", type=str, metavar="D", default="auto",
        help="device used for heavy computations, e.g. cpu or cuda")
    parser.add_argument("--rand_init", action="store_true",
        help="initialize world from random noise")
    parser.add_argument("--no_preview", action="store_true",
        help="disable video preview while rendering")
    parser.add_argument("--preserve_out_dir", action="store_true",
        help="preserve output png directory after compiling video")
    return parser.parse_args()

version = "1.0"
author = "Santiago Benoit"
title = r"""
 _____    _   _    _____    _____    _____    _   _    _____    _____      _   
/_____/  |_| |_|  |_____\  |____ /  |_____\  | \ |_|  /_ _ _\  \ ___ /    /_\  
\_____\  |_|_|_|  |_____/  |____|   |_____/  |_|o|_|  |_|_|_|   \ _ /    /___\ 
/_____/  \_____/  |_|      |_____\  |_____\  |_| \_|  \_____/    \_/    /_____\
                                                                               
                        Convolutional Audio Visualizer                         
                                  Version 1.0                                  
                           (c) 2022 Santiago Benoit                            
"""
effects_help = """add effects to output video
    identity: no effect
    invert: invert colors
    sobel: apply Sobel filter for edge emphasis
    glow: apply glow effect using Sobel filter overlay
    grayscale: convert colors to grayscale
    saturate: increase saturation
    desaturate: decrease saturation
    normalize: stretch contrast using top and bottom percentile
    sblur: apply spacial blur
    mblur: apply motion blur
    hmirror: mirror horizontally
    vmirror: mirror vertically"""

def main():
    global preview_open
    warnings.filterwarnings("ignore")
    mpl.use("TkCairo")
    torch.set_grad_enabled(False)
    print(title)
    args = get_args()
    preview = not args.no_preview
    if args.seed_str is not None:
        if args.seed_int is not None:
            raise ValueError("seed_str and seed_int both specified")
        else:
            print("Seed string:", args.seed_str)
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[-16:], 16)
            torch.manual_seed(seed)
    elif args.seed_int is not None:
        seed = args.seed_int
        torch.manual_seed(seed)
    else:
        seed = torch.seed()
    print("Random seed:", seed)
    print("Initializing model...")
    device = get_device(args.device)
    print("Using device", device.type)
    out_dim = sum(3 * 3 * size * size + 12 for size in args.filter_sizes)
    model = create_model(out_dim, device)
    if args.rand_init:
        world = torch.rand(3, args.video_dims[1], args.video_dims[0]).to(device) * 2 - 1
    else:
        world = torch.zeros(3, args.video_dims[1], args.video_dims[0]).to(device)
    delta = []
    effects = init_effects(args.effects, args)
    if args.audio_file is None:
        print("No audio file specified. Using silent mode.")
        if args.video_length is None:
            raise ValueError("video_length must be specified in silent mode")
        else:
            features = None
            total_steps = args.video_length
    else:
        print("Analyzing audio...")
        pcm, _ = librosa.load(args.audio_file, sr=args.sr)
        features = audio_features(pcm, args).to(device)
        total_steps = features.size(0)
        del pcm
    print("Rendering frames...")
    if os.path.isdir(args.out_dir):
        yn = input(f"Directory {args.out_dir} exists. Overwrite? (y/n) ")
        if yn.lower() == "y":
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)
        else:
            abort()
    else:
        os.mkdir(args.out_dir)
    digits = int(math.log10(total_steps)) + 1
    if preview:
        fx = apply_effects(world, effects)
        img = to_img(fx)
        dpi = mpl.rcParams["figure.dpi"]
        figsize = (args.video_dims[0] / dpi, args.video_dims[1] / dpi)
        fig = plt.figure("Supernova", figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        imshow = ax.imshow(img, interpolation="none")
        preview_open = True
        fig.canvas.mpl_connect("close_event", close_preview)
        fig.show()
    for global_step in tqdm.tqdm(range(total_steps)):
        world = step(world, model, delta, features, global_step, args)
        fx = apply_effects(world, effects)
        img = to_img(fx)
        if preview and preview_open:
            imshow.set_data(img)
            fig.canvas.draw()
            fig.canvas.flush_events()
        img.save(os.path.join(args.out_dir, f"frame_{str(global_step).zfill(digits)}.png"))
    if preview and preview_open:
        fig.close()
    print("Compiling video...")
    if args.audio_file is None:
        cmd = ["ffmpeg", "-framerate", str(args.fr), "-i", os.path.join(args.out_dir, f"frame_%0{digits}d.png"), "-c:v", "libx265", "-x265-params", "lossless=1", args.out_file]
    else:
        cmd = ["ffmpeg", "-framerate", str(args.fr), "-i", os.path.join(args.out_dir, f"frame_%0{digits}d.png"), "-i", args.audio_file, "-map", "0:v", "-map", "1:a",  "-c:v", "libx265", "-x265-params", "lossless=1", "-c:a", "copy", "-shortest", args.out_file]
    subprocess.run(cmd, check=True)
    if not args.preserve_out_dir:
        print("Cleaning up...")
        shutil.rmtree(args.out_dir)
    finish()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        abort()
