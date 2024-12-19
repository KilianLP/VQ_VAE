import torch.optim as optim
import wandb
import numpy as np
from models import Autoencoder
from dataset import dataset
import torch as T
from torch.utils.data import DataLoader,random_split

# hyperparameters
batch_size = 2
lr = 2e-3
epochs = 60
accumulation_step = 1
n_resnet_blocks = 3
channels = [1,64,128]
n_embeddings = 128
data_directory = '/content/drive/MyDrive/Data/all_spectrograms.csv'
model_parameters_directory = ''
# prepare data

data = np.loadtxt(data_directory, delimiter=',', dtype=np.float32)
tensor = T.tensor(data).view(-1, 1, 2048, 2048)
data = T.log(tensor)
data_min = data.min()
data_max = data.max()

normalized_data = 2 * (data - data_min) / (data_max - data_min) - 1

dataset = Dataset(normalized_data)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# prepare model

autoencoder = Autoencoder(n_embeddings,channels,n_resnet_block)
device = T.device("cuda" if T.cuda.is_available() else "cpu")
autoencoder.to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr = lr)

# train
wandb.init(project="VQ_VAE")

for _ in range(epochs):
  total_loss = 0
  psnr = 0
  autoencoder.train()
  idx = 0
  for batch in dataloader:
    batch = batch.to(device)

    output,loss = autoencoder(batch)
    mse_loss = loss_fn(output,batch)
    loss_value = mse_loss + loss
    loss_value.backward()

    if idx % accumulation_step == 0:
      optimizer.step()
      optimizer.zero_grad()

    idx += 1
    total_loss = loss_value.item()
    psnr = 10*T.log10(1/(T.tensor(mse_loss.item())))

    wandb.log({"total_loss": total_loss})
    wandb.log({"mse_loss": mse_loss.item()})
    wandb.log({"psnr": psnr})

  autoencoder.eval()
  total_loss = 0
  psnr = 0
  with T.no_grad():
    for batch in dataloader_test:
      batch = batch.to(device)
      output,_ = autoencoder(batch)
      loss_value = loss_fn(output,batch)
      total_loss += loss_value.item()
      psnr = 10*T.log10(1/(T.tensor(loss_value.item())))

      wandb.log({"test_mse_loss": loss_value.item()})
      wandb.log({"test_psnr": psnr})


T.save(autoencoder.state_dict(), model_parameters_directory + 'VQ_VAE_parameters.pth')
