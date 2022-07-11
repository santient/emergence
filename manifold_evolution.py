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
import tqdm

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
    coords = 5 * torch.stack(torch.meshgrid(torch.linspace(-1, 1, world.size(1)), torch.linspace(-1, 1, world.size(2)), indexing='ij'), dim=-1).cuda()
    with torch.no_grad():
        filters = model(coords)
    # print(filters.var())
    # filters = filters / filters.var(dim=2, keepdim=True)
    return filters

def create_model(out_dim):
    model = Sequential(
        Linear(2, out_dim),
        ReLU(),
        Linear(out_dim, out_dim)
        ).cuda()
    for p in model.parameters():
        if p.dim() > 1:
            xavier_normal_(p, 2)
        else:
            normal_(p, 0, 0.25)
    return model

def step(world, model, delta, global_step):
    filters = get_filters(world, model)
    world_out = world
    start = 0
    # for idx, size in enumerate(sizes):
    #     end = start + 3 * 3 * size * size
    #     # scale = filters[:, :, start:start + 1]
    #     # scale = 1
    #     fil = (filters[:, :, start:end]).view(world.size(1), world.size(2), 3, 3, size, size)
    #     start = end
    #     pad = size // 2
    #     if pad > 0:
    #         # padded = circular_pad(world_out, pad)
    #         padded = F.pad(world_out, (pad, pad, pad, pad), mode='constant')
    #     else:
    #         padded = world_out
    #     unfold = padded.unfold(1, size, 1).unfold(2, size, 1)
    #     world_out = (fil.permute(2, 0, 1, 3, 4, 5) * unfold.unsqueeze(3)).sum(dim=(3, 4, 5))
    #     # print(world_out.shape)
    #     # result = torch.empty_like(world_out)
    #     # for i in range(world_out.size(1)):
    #     #     for j in range(world_out.size(2)):
    #     #         # print(i, j)
    #     #         result[:, i, j] = (fil[i, j].view(3, 3, size * size) @ padded[:, i:i + size, j:j + size].reshape(3, size * size, 1)).sum(dim=(1, 2))
    #     # world_out = result
    #     if idx < len(sizes) - 1:
    #         world_out = torch.relu(world_out)
    for idx, size in enumerate(sizes):
        end = start + 3 + 3 * 3 * size * size
        # scale = filters[:, :, start:start + 1]
        # scale = 1
        bias = filters[:, :, start:start + 3].view(world.size(1), world.size(2), 3)
        fil = filters[:, :, start + 3:end].view(world.size(1), world.size(2), 3, 3, size, size)
        start = end
        pad = size // 2
        if pad > 0:
            # padded = circular_pad(world_out, pad)
            padded = F.pad(world_out, (pad, pad, pad, pad), mode='reflect')
        else:
            padded = world_out
        unfold = padded.unfold(1, size, 1).unfold(2, size, 1)
        world_out = (fil.permute(2, 0, 1, 3, 4, 5) * unfold.unsqueeze(3)).sum(dim=(3, 4, 5))
        if add_bias:
            world_out = world_out + bias.permute(2, 0, 1)
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
    if global_step % 1800 == 0:
        delta.clear()
        waypoint = create_model(out_dim)
        for p, w in zip(model.parameters(), waypoint.parameters()):
            with torch.no_grad():
                delta.append((w.data - p.data) / 1800)
    for p, d in zip(model.parameters(), delta):
        with torch.no_grad():
            p.data = p.data + d
    return world_out

def to_img(world):
    img = world.permute(1, 2, 0).cpu().numpy()
    # img = img - img.min()
    # img = img / img.max()
    img = img + 1
    img = img / 2
    img = img * 255
    img = np.round(img).astype("uint8")
    return Image.fromarray(img)

dims = (720, 720)
sizes = (1, 3, 5, 7)
out_dim = sum(3 + 3 * 3 * size * size for size in sizes)
visualize = True
add_bias = False
out_dir = 'out/'
total_steps = 18000

if __name__ == '__main__':
    if len(sys.argv) == 3:
        if sys.argv[1] == '-s':
            seed_str = sys.argv[2]
            print("Seed string:", seed_str)
            seed = int(hashlib.md5(seed_str.encode()).hexdigest()[-16:], 16)
        elif sys.argv[1] == '-i':
            seed = int(sys.argv[2])
        else:
            raise ValueError('Invalid arguments.')
        print("Seed integer:", seed)
        torch.manual_seed(seed)
    elif len(sys.argv) == 1:
        seed = torch.seed()
        print("Using random seed:", seed)
    else:
        raise ValueError('Invalid arguments.')
    model = create_model(out_dim)
    world = torch.rand(3, dims[0], dims[1]).cuda() * 2 - 1
    # world = torch.zeros(3, dims[0], dims[1]).cuda()
    delta = []
    # filters = get_filters(world, model)
    # coords = 100 * torch.stack(torch.meshgrid(torch.linspace(-1, 1, world.size(1)), torch.linspace(-1, 1, world.size(2)), indexing='ij'), dim=-1).cuda()
    # with torch.no_grad():
    #     filters = model(coords)
    #     pca, _, _ = torch.pca_lowrank(filters.view(-1, out_dim), q=3, center=True)
    #     pca = pca.view(world.size(1), world.size(2), 3).permute(2, 0, 1)
    if visualize:
        global_step = 0
        root = tk.Tk()
        root.title("Emergence")
        canvas = tk.Canvas(root, width=dims[0], height=dims[1])
        canvas.pack()
        running = True
        def close():
            global running
            running = False
        def update():
            global world
            global model
            # global filters
            global delta
            global global_step
            global canvas
            global root
            global running
            while running:
                # time.sleep(1 / 60)
                img = to_img(world)
                img_tk = ImageTk.PhotoImage(img)
                canvas.delete("all")
                canvas.create_image(dims[0], dims[1], anchor="se", image=img_tk)
                world = step(world, model, delta, global_step)
                global_step += 1
                root.update()
            root.destroy()
        root.after(0, update)
        os.system('xset r off')
        root.protocol("WM_DELETE_WINDOW", close)
        root.mainloop()
        os.system('xset r on')
    else:
        for global_step in tqdm.tqdm(range(total_steps)):
            img = to_img(world)
            img.save(os.path.join(out_dir, f'{global_step:05}.png'))
            world = step(world, model, delta, global_step)
