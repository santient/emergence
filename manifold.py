import os
import sys
import hashlib
import time
import numpy as np
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.nn.init import normal_, xavier_normal_
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageTk

def circular_pad(arr, pad):
    padded = F.pad(arr, (pad, pad, pad, pad))
    padded[..., :pad, pad:-pad] = arr[..., -pad:, :]
    padded[..., -pad:, pad:-pad] = arr[..., :pad, :]
    padded[..., pad:-pad, :pad] = arr[..., :, -pad:]
    padded[..., pad:-pad, -pad:] = arr[..., :, :pad]
    padded[..., :pad, :pad] = arr[..., -pad:, -pad:]
    padded[..., :pad, -pad:] = arr[..., -pad:, :pad]
    padded[..., -pad:, :pad] = arr[..., :pad, -pad:]
    padded[..., -pad:, -pad:] = arr[..., :pad, :pad]
    return padded

def get_filters(world, model):
    coords = torch.stack(torch.meshgrid(torch.linspace(-1, 1, world.size(1)), torch.linspace(-1, 1, world.size(2)), indexing='ij'), dim=-1).cuda()
    with torch.no_grad():
        filters = model(coords * 5)
    # print(filters.var())
    # filters = filters / filters.var(dim=2, keepdim=True)
    return filters

def step(world, filters, sizes):
    world_out = world
    start = 0
    for idx, size in enumerate(sizes):
        end = start + 3 * 3 * size * size
        # scale = filters[:, :, start:start + 1]
        # scale = 1
        fil = (filters[:, :, start:end]).view(world.size(1), world.size(2), 3, 3, size, size)
        start = end
        pad = size // 2
        if pad > 0:
            # padded = circular_pad(world_out, pad)
            padded = F.pad(world_out, (pad, pad, pad, pad), mode='constant')
        else:
            padded = world_out
        unfold = padded.unfold(1, size, 1).unfold(2, size, 1)
        world_out = (fil.permute(2, 0, 1, 3, 4, 5) * unfold.unsqueeze(3)).sum(dim=(3, 4, 5))
        # print(world_out.shape)
        # result = torch.empty_like(world_out)
        # for i in range(world_out.size(1)):
        #     for j in range(world_out.size(2)):
        #         # print(i, j)
        #         result[:, i, j] = (fil[i, j].view(3, 3, size * size) @ padded[:, i:i + size, j:j + size].reshape(3, size * size, 1)).sum(dim=(1, 2))
        # world_out = result
        if idx < len(sizes) - 1:
            world_out = torch.relu(world_out)
    world_out = torch.tanh(world_out)
    return world_out

def to_img(world):
    img = world.permute(1, 2, 0).cpu().numpy()
    # img = img - img.min()
    # img = img / img.max()
    img = img + 1
    img = img / 2
    img = img * 255
    img = np.round(img).astype("uint8")
    return ImageTk.PhotoImage(Image.fromarray(img))

if __name__ == '__main__':
    if len(sys.argv) > 1:
        seed_str = sys.argv[1]
        print("Seed string:", seed_str)
        seed = int(hashlib.md5(seed_str.encode()).hexdigest()[-16:], 16)
        print("Seed integer:", seed)
        torch.manual_seed(seed)
    else:
        seed = torch.seed()
        print("Using random seed:", seed)
    world = torch.rand(3, 720, 720).cuda() * 2 - 1
    sizes = (1, 3, 5, 7)
    out_dim = sum(3 * 3 * size * size for size in sizes)
    model = Sequential(
        Linear(2, out_dim),
        ReLU(),
        Linear(out_dim, out_dim)
        ).cuda()
    for p in model.parameters():
        # if p.dim() == 1:
        #     normal_(p, 0, 0.2)
        if p.dim() > 1:
            xavier_normal_(p, 2)
        else:
            normal_(p, 0, 0.2)
    filters = get_filters(world, model)
    # coords = 100 * torch.stack(torch.meshgrid(torch.linspace(-1, 1, world.size(1)), torch.linspace(-1, 1, world.size(2)), indexing='ij'), dim=-1).cuda()
    # with torch.no_grad():
    #     filters = model(coords)
    #     pca, _, _ = torch.pca_lowrank(filters.view(-1, out_dim), q=3, center=True)
    #     pca = pca.view(world.size(1), world.size(2), 3).permute(2, 0, 1)
    root = tk.Tk()
    root.title("Emergence")
    canvas = tk.Canvas(root, width=720, height=720)
    canvas.pack()
    running = True
    def close():
        global running
        running = False
    def update():
        global world
        global filters
        global sizes
        global canvas
        global root
        global running
        while running:
            time.sleep(1/60)
            img = to_img(world)
            canvas.delete("all")
            canvas.create_image(720, 720, anchor="se", image=img)
            world = step(world, filters, sizes)
            root.update()
        root.destroy()
    root.after(0, update)
    os.system('xset r off')
    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    os.system('xset r on')
