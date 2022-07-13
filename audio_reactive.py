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
# import tkinter as tk
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import tqdm
import librosa
import subprocess
import warnings

warnings.filterwarnings('ignore')

# def circular_pad(arr, pad):
#     padded = F.pad(arr, (pad, pad, pad, pad))
#     padded[..., :pad, pad:-pad] = arr[..., -pad:, :]
#     padded[..., -pad:, pad:-pad] = arr[..., :pad, :]
#     padded[..., pad:-pad, :pad] = arr[..., :, -pad:]
#     padded[..., pad:-pad, -pad:] = arr[..., :, :pad]
#     padded[..., :pad, :pad] = arr[..., -pad:, -pad:]
#     padded[..., :pad, -pad:] = arr[..., -pad:, :pad]
#     padded[..., -pad:, :pad] = arr[..., :pad, -pad:]
#     padded[..., -pad:, -pad:] = arr[..., :pad, :pad]
#     return padded

def get_filters(world, model):
    coords = 5 * torch.stack(torch.meshgrid(torch.linspace(-1, 1, world.size(1)), torch.linspace(-1, 1, world.size(2)), indexing='ij'), dim=-1).to(model.device)
    with torch.no_grad():
        filters = model(coords)
    # print(filters.var())
    # filters = filters / filters.var(dim=2, keepdim=True)
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
    features = np.concatenate((np.zeros((features.shape[0], 1)), features), axis=1)
    features = features.transpose(1, 0).copy()
    return torch.from_numpy(features)

def step(world, model, delta, features, global_step, args):
    filters = get_filters(world, model)
    world_out = world
    start = 0
    for idx, size in enumerate(args.filter_sizes):
        if features is not None:
            scale = args.sensitivity * (features[global_step, 0] / 2 + features[global_step, idx + 1] * 2)
        else:
            scale = 1
        end = start + 3 + 3 * 3 * size * size
        # scale = filters[:, :, start:start + 1]
        # scale = 1
        if features is not None:
            bias = filters[:, :, start:start + 3].view(world.size(1), world.size(2), 3)
        fil = scale * filters[:, :, start + 3:end].view(world.size(1), world.size(2), 3, 3, size, size)
        start = end
        pad = size // 2
        if pad > 0:
            # padded = circular_pad(world_out, pad)
            padded = F.pad(world_out, (pad, pad, pad, pad), mode='reflect')
        else:
            padded = world_out
        unfold = padded.unfold(1, size, 1).unfold(2, size, 1)
        world_out = (fil.permute(2, 0, 1, 3, 4, 5) * unfold.unsqueeze(3)).sum(dim=(3, 4, 5))
        if features is not None:
            world_out = world_out + bias.permute(2, 0, 1)
        # print(world_out.shape)
        # result = torch.empty_like(world_out)
        # for i in range(world_out.size(1)):
        #     for j in range(world_out.size(2)):
        #         # print(i, j)
        #         result[:, i, j] = (fil[i, j].view(3, 3, size * size) @ padded[:, i:i + size, j:j + size].reshape(3, size * size, 1)).sum(dim=(1, 2))
        # world_out = result
        if idx < len(args.filter_sizes) - 1:
            world_out = torch.relu(world_out)
    world_out = torch.tanh(world_out)
    if args.interval > 0 and global_step % args.interval == 0:
        delta.clear()
        waypoint = create_model(model.out_dim, model.device)
        for p, w in zip(model.parameters(), waypoint.parameters()):
            with torch.no_grad():
                delta.append((w.data - p.data) / args.interval)
    if delta:
        for p, d in zip(model.parameters(), delta):
            with torch.no_grad():
                p.data = p.data + d
    return world_out

def to_img(world):
    img = (world + 1) / 2
    img = img * 255
    img = img.permute(1, 2, 0).cpu().numpy()
    # img = img - img.min()
    # img = img / img.max()
    img = np.round(img).astype("uint8")
    # return img
    return Image.fromarray(img)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_file', type=str, required=True,
        help='output video file path')
    parser.add_argument('--audio_file', type=str,
        help='input audio file path')
    parser.add_argument('--out_dir', type=str, default='out',
        help='output directory for video frames')
    parser.add_argument('--seed_str', type=str,
        help='seed string (hashed into 64-bit integer)')
    parser.add_argument('--seed_int', type=int,
        help='64-bit seed integer')
    parser.add_argument('--video_dims', type=int, nargs=2, default=(720, 720),
        help='output video dimensions')
    parser.add_argument('--video_length', type=int,
        help='manually specify video length in frames')
    parser.add_argument('--filter_sizes', type=int, nargs='+', default=[1, 3, 5, 7],
        help='convolutional filter sizes')
    parser.add_argument('--fr', type=int, default=30,
        help='video frame rate')
    parser.add_argument('--sr', type=int, default=22050,
        help='audio sample rate')
    parser.add_argument('--sensitivity', type=float, default=1.0,
        help='audio reactive sensitivity')
    parser.add_argument('--interval', type=int, default=1800,
        help='evolution interval for model weights (0 for no evolution)')
    # TODO implement effects
    # parser.add_argument('--effects', type=str, nargs='*', default=[],
        # help='add effects to output video: invert, gradient, grayscale, blur, hmirror, vmirror')
    parser.add_argument('--device', type=str,
        help='device used for heavy computations, e.g. cpu or cuda')
    parser.add_argument('--visualize', action='store_true',
        help='visualize video while rendering')
    parser.add_argument('--preserve_out_dir', action='store_true',
        help='preserve output directory after compiling video')
    return parser.parse_args()

# dims = (720, 720)
# sizes = (1, 3, 5, 7)
# out_dim = sum(3 + 3 * 3 * size * size for size in sizes)
# out_dir = 'out/'
# sr = 22050
# fr = 30
# visualize = False
# cleanup = False
# interval = 1800
version = "1.0"
author = "Santiago Benoit"
title = r"""
_________                   _____________  __
__  ____/_______________   ____  ____/_  |/ /
_  /    _  __ \_  __ \_ | / /_  /_   __    / 
/ /___  / /_/ /  / / /_ |/ /_  __/   _    |  
\____/  \____//_/ /_/_____/ /_/      /_/|_|  
                                             
           Neural Audio Visualizer           
                 Version 1.0                 
          (c) 2022 Santiago Benoit           
"""

if __name__ == '__main__':
    print(title)
    args = get_args()
    # if len(sys.argv) == 5:
    #     if sys.argv[3] == '-s':
    #         seed_str = sys.argv[4]
    #         print("Seed string:", seed_str)
    #         seed = int(hashlib.md5(seed_str.encode()).hexdigest()[-16:], 16)
    #     elif sys.argv[3] == '-i':
    #         seed = int(sys.argv[4])
    #     else:
    #         raise ValueError('Invalid arguments.')
    #     print("Seed integer:", seed)
    #     torch.manual_seed(seed)
    # elif len(sys.argv) == 3:
    #     seed = torch.seed()
    #     print("Using random seed:", seed)
    # else:
    #     raise ValueError('Invalid arguments.')
    # audio_file = sys.argv[1]
    # out_file = sys.argv[2]
    if args.seed_str is not None:
        if args.seed_int is not None:
            raise ValueError('seed_str and seed_int both specified')
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
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print("Using device", device.type)
    out_dim = sum(3 + 3 * 3 * size * size for size in args.filter_sizes)
    model = create_model(out_dim, device)
    if args.audio_file is None:
        world = torch.rand(3, args.video_dims[0], args.video_dims[1]).to(device) * 2 - 1
    else:
        world = torch.zeros(3, args.video_dims[0], args.video_dims[1]).to(device)
    delta = []
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
    # filters = get_filters(world, model)
    # coords = 100 * torch.stack(torch.meshgrid(torch.linspace(-1, 1, world.size(1)), torch.linspace(-1, 1, world.size(2)), indexing='ij'), dim=-1).cuda()
    # with torch.no_grad():
    #     filters = model(coords)
    #     pca, _, _ = torch.pca_lowrank(filters.view(-1, out_dim), q=3, center=True)
    #     pca = pca.view(world.size(1), world.size(2), 3).permute(2, 0, 1)
    # if visualize:
    #     global_step = 0
    #     root = tk.Tk()
    #     root.title("Emergence")
    #     canvas = tk.Canvas(root, width=dims[0], height=dims[1])
    #     canvas.pack()
    #     running = True
    #     def close():
    #         global running
    #         running = False
    #     def update():
    #         global world
    #         global model
    #         # global filters
    #         global delta
    #         global global_step
    #         global canvas
    #         global root
    #         global running
    #         while running:
    #             time.sleep(1 / 60)
    #             img = to_img(world)
    #             img_tk = ImageTk.PhotoImage(img)
    #             canvas.delete("all")
    #             canvas.create_image(dims[0], dims[1], anchor="se", image=img_tk)
    #             world = step(world, model, delta, global_step)
    #             global_step += 1
    #             root.update()
    #         root.destroy()
    #     root.after(0, update)
    #     os.system('xset r off')
    #     root.protocol("WM_DELETE_WINDOW", close)
    #     root.mainloop()
    #     os.system('xset r on')
    # else:
    print("Rendering frames...")
    digits = int(math.log10(total_steps)) + 1
    if args.visualize:
        dpi = mpl.rcParams['figure.dpi']
        figsize = (args.video_dims[0] / dpi, args.video_dims[1] / dpi)
        fig = plt.figure("ConvFX", figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        imshow = ax.imshow(to_img(world), interpolation='none')
        # fig.tight_layout()
        fig.show()
    os.mkdir(args.out_dir)
    for global_step in tqdm.tqdm(range(total_steps)):
        world = step(world, model, delta, features, global_step, args)
        img = to_img(world)
        if args.visualize:
            imshow.set_data(img)
            fig.canvas.draw()
            fig.canvas.flush_events()
        img.save(os.path.join(args.out_dir, f'frame_{str(global_step).zfill(digits)}.png'))
    print("Compiling video...")
    if args.audio_file is None:
        cmd = ['ffmpeg', '-framerate', str(args.fr), '-i', os.path.join(args.out_dir, f'frame_%0{digits}d.png'), '-c:v', 'libx265', '-x265-params', 'lossless=1', args.out_file]
    else:
        cmd = ['ffmpeg', '-framerate', str(args.fr), '-i', os.path.join(args.out_dir, f'frame_%0{digits}d.png'), '-i', args.audio_file, '-map', '0:v', '-map', '1:a',  '-c:v', 'libx265', '-x265-params', 'lossless=1', '-c:a', 'copy', '-shortest', args.out_file]
    subprocess.run(cmd, check=True)
    if not args.preserve_out_dir:
        print("Cleaning up...")
        shutil.rmtree(args.out_dir)
    print("Done!")
