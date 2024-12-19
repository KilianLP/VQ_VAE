import torch.optim as optim
import wandb
import numpy as np
from models import Autoencoder
import torch

# prepare data

data = np.loadtxt('/content/drive/MyDrive/Data/all_spectrograms.csv', delimiter=',', dtype=np.float32)
tensor = torch.tensor(data).view(-1, 1, 2048, 2048)
print(tensor.size())
data = T.log(tensor)
data_min = data.min()
data_max = data.max()

normalized_data = 2 * (data - data_min) / (data_max - data_min) - 1



autoencoder = Autoencoder(128,[1,64,128],3)
device = T.device("cuda" if T.cuda.is_available() else "cpu")
autoencoder.to(device)

accumulation_step = 1

loss_fn = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr = 2e-3)

wandb.init(project="VQ_VAE_Spectro_L")

for _ in range(60):
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

