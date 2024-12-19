import random
import numpy as np
import matplotlib.pyplot as plt
from models import Autoencoder
import torch

# hyperparameters
n_resnet_blocks = 3
channels = [1,64,128]
n_embeddings = 128
model_parameters_directory = ''

autoencoder = Autoencoder(n_embeddings,channels,n_resnet_block)
autoencoder.load_state_dict(torch.load(model_parameters_directory))

images = []
for i in range(5):
    k = random.randrange(1, len(test_dataset))
    image = test_dataset.__getitem__(k)
    image = image.permute(1,2,0)
    image = (image + 1) / 2 * (data_max - data_min) + data_min

    image = image.numpy()
    images.append(image)

    autoencoder.eval()
    image = test_dataset.__getitem__(k)
    image, _= autoencoder(image.unsqueeze(0).to(device))
    image = image.squeeze(0)
    image = image.cpu()
    image = image.permute(1,2,0)
    image = (image + 1) / 2 * (data_max - data_min) + data_min
    image = image.detach().numpy()

    images.append(image)


fig, axs = plt.subplots(5,2 , figsize=(6, 15))


for i, ax in enumerate(axs.flat):
    ax.imshow(images[i], aspect='auto', cmap='inferno', origin='lower')
    if i % 2 == 0:
        ax.set_title(f'Original Image {i//2+1}')
    else:
      ax.set_title(f'Reconstructed Image {i//2 + 1}')
    ax.axis('off')

plt.tight_layout()
plt.show()
