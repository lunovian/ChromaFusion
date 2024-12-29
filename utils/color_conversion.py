import numpy as np
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
import torch

def lab_to_rgb(L, AB):
    L = L.squeeze(1).cpu().numpy()
    AB = AB.permute(0, 2, 3, 1).cpu().numpy()
    L = L * 100
    AB = (AB - 0.5) * 128 * 2
    images = np.zeros((L.shape[0], L.shape[1], L.shape[2], 3))
    images[:, :, :, 0] = L
    images[:, :, :, 1:] = AB
    images_rgb = [lab2rgb(image) for image in images]
    return np.array(images_rgb)

def display_progress(cond, real, fake, current_epoch=0, figsize=(20, 15)):
    cond = cond.detach().cpu().unsqueeze(0)
    real = real.detach().cpu().unsqueeze(0)
    fake = fake.detach().cpu().unsqueeze(0)
    images = [cond, real, fake]
    titles = ['input', 'real', 'generated']
    print(f'Epoch: {current_epoch}')
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for idx, img in enumerate(images):
        if idx == 0:
            ab = torch.zeros_like(img).repeat(1, 2, 1, 1)
            img = torch.cat([img * 100, ab], dim=1)
            img = img.permute(0, 2, 3, 1).numpy()[0]
            imgan = lab2rgb(img)
        else:
            imgan = lab_to_rgb(images[0], img)[0]
        ax[idx].imshow(imgan)
        ax[idx].axis("off")
        ax[idx].set_title(titles[idx])
    plt.show()