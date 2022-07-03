import os
import sys
import hashlib
import time
import numpy as np
import torch
import torch.nn.functional as F
import tkinter as tk
from PIL import Image, ImageTk

def reduce_mean(arr, sz):
    chan = arr.size(0)
    block = arr.size(1) // sz
    return arr.view(chan, sz, sz, block, block).mean(dim=(2, 3))

def reduce_sum(arr, sz):
    chan = arr.size(0)
    block = arr.size(1) // sz
    return arr.view(chan, sz, sz, block, block).sum(dim=(2, 3))

def reduce_det(arr, sz):
    chan = arr.size(0)
    block = arr.size(1) // sz
    return torch.det(arr.view(chan, sz, sz, block, block))

def reduce_eig(arr, sz):
    chan = arr.size(0)
    block = arr.size(1) // sz
    ev = torch.linalg.eigvals(arr.view(chan, sz, sz, block, block))
    return ev.permute(0, 3, 1, 2)[:, :3, :, :]

def reduce_var(arr, sz):
    chan = arr.size(0)
    block = arr.size(1) // sz
    return arr.view(chan, sz, sz, block, block).var(dim=(3,4)).log()

def complex_conv2d(arr, fil):
    a = arr.real
    b = arr.imag
    c = fil.real
    d = fil.imag
    r = F.conv2d(a, c, padding='same') - F.conv2d(b, d, padding='same')
    i = F.conv2d(a + b, c + d, padding='same') - r
    return torch.complex(r, i)

# def step(arr, n):
#     out = arr
#     for i in range(1, n + 1):
#         # fil = reduce_det(arr, i).unsqueeze(1)
#         # fil = torch.randn(3, 3, i, i)
#         # fil += 1
#         fil = reduce_var(arr, i).unsqueeze(1)
#         out = F.conv2d(out.unsqueeze(0), fil, padding='same', groups=3).squeeze(0)
#     out = F.tanh(out)
#     print(out.abs().min(), out.abs().max())
#     return out

def step(arr, fil, mom):
    arr_out = arr.unsqueeze(0)
    fil_out = []
    mom_out = []
    var = arr.var(dim=(1, 2)).mean()
    # print(var)
    coeff = torch.sigmoid(-((var * 100).log()))
    # print(coeff)
    # coeff = 0
    for i, (f, m) in enumerate(zip(fil, mom)):
        # print(f.shape, m.shape, "next")
        arr_out = F.conv2d(arr_out, f, padding='same')
        if i < len(fil) - 1:
            arr_out = torch.relu(arr_out)
        f_out = f + (m - f) * coeff
        f_out = f_out / f_out.var()
        fil_out.append(f_out)
        m_out = m + (torch.randn_like(m) - m) * 0.01
        mom_out.append(m_out)
    arr_out = arr_out + (torch.randn_like(arr_out) - arr_out) * coeff
    arr_out = torch.tanh(arr_out)
    return arr_out.squeeze(0), fil_out, mom_out

def toImg(arr):
    arr = arr.permute(1, 2, 0).numpy()
    # arr = arr - arr.min()
    # arr = arr / arr.max()
    arr = arr + 1
    arr = arr / 2
    arr = arr * 255
    arr = np.round(arr).astype("uint8")
    return ImageTk.PhotoImage(Image.fromarray(arr))

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
    arr = torch.rand(3, 720, 720) * 2 - 1
    # seed = arr
    # arr = F.tanh(arr)
    n = 4
    fil = []
    for i in range(n):
        s = 2 * i + 1
        f = torch.randn(3, 3, s, s)
        f = f / f.var()
        fil.append(f)
    mom = []
    for i in range(n):
        s = 2 * i + 1
        m = torch.randn(3, 3, s, s)
        mom.append(m)
    root = tk.Tk()
    root.title("Emergence")
    canvas = tk.Canvas(root, width=720, height=720)
    canvas.pack()
    def update():
        global arr
        global fil
        global mom
        global canvas
        global root
        while True:
            time.sleep(1/60)
            img = toImg(arr)
            canvas.delete("all")
            canvas.create_image(720, 720, anchor="se", image=img)
            arr, fil, mom = step(arr, fil, mom)
            root.update()
    root.after(0, update)
    os.system('xset r off')
    root.mainloop()
    os.system('xset r on')
