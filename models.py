import torch as T
import torch.nn as nn


class EfficientVectorQuantizer(nn.Module):
  def __init__(self,emb_dim,n_emb,beta=0.25):
    super().__init__()

    self.embeddings = nn.Embedding(n_emb,emb_dim)
    self.embeddings.weight.data.uniform_(-1.0 / n_emb, 1.0 / n_emb)
    self.beta = beta

  def forward(self,x):

    b,c,h,w = x.size()
    flatten_x = x.permute(0,2,3,1).contiguous()
    flatten_x = flatten_x.view(b*h*w,c)
    
    dist = T.sum(flatten_x**2,dim = -1,keepdim = True) + T.sum(self.embeddings.weight**2, dim = -1).unsqueeze(0) - 2 * flatten_x @ self.embeddings.weight.T
    
    idx = dist.argmin(dim = -1)
    emb_x = self.embeddings(idx)

    emb_x = emb_x.view(b,h,w,c)
    emb_x = emb_x.permute(0,3,1,2)

    loss = ((x - emb_x.detach())**2).mean() + self.beta * ((x.detach() - emb_x)**2).mean()

    emb_x = emb_x.detach() + x - x.detach()

    return emb_x, loss



class Encoder(nn.Module):
  def __init__(self,channels,n_res):
    super().__init__()

    self.net = nn.Sequential()
    for c in range(len(channels)-1):
      self.net.add_module(f"enc_block_{c}",EncoderBlock(channels[c],channels[c+1]))

    self.net.add_module(f"conv_block_{1}",nn.Conv2d(channels[-1],channels[-1],3,stride = 1, padding = 1))

    for i in range(n_res):
      self.net.add_module(f"res_block_{i}",ResidualLayer(channels[-1],channels[-1],channels[-1]))

  def forward(self,x):
    return self.net(x)

class EncoderBlock(nn.Module):
  def __init__(self,input_channel,out_channel):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(input_channel,out_channel,4,stride = 2, padding = 1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )

  def forward (self,x):
    return self.net(x)

class DecoderBlock(nn.Module):
  def __init__(self,input_channel,out_channel):
    super().__init__()

    self.net = nn.Sequential(
        nn.ConvTranspose2d(input_channel,out_channel,3,stride = 2, padding = 1, output_padding = 1),
        nn.BatchNorm2d(out_channel),
        )

  def forward (self,x):
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self,channels,n_res):
    super().__init__()

    self.net = nn.Sequential()
    for c in range(len(channels)-1):
      if c ==1:
        for i in range(n_res):
          self.net.add_module(f"res_block_{c+i}",ResidualLayer(channels[c],channels[c],channels[c]))
      if c != 0:
        self.net.add_module(f"relu_block_{c}",nn.ReLU())

      self.net.add_module(f"dec_block_{c}",DecoderBlock(channels[c],channels[c+1]))


  def forward(self,x):
    return self.net(x)


class ResidualLayer(nn.Module):
  def __init__(self,input_channel,h_channel,out_channel):
    super().__init__()

    self.net = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(input_channel,h_channel,3,stride = 1, padding = 1,bias=False),
        nn.BatchNorm2d(h_channel),
        nn.ReLU(),
        nn.Conv2d(h_channel,out_channel,3,stride = 1, padding = 1,bias=False),
        nn.BatchNorm2d(out_channel),
    )

  def forward(self,x):
    x = x + self.net(x)

    return x


class Autoencoder(nn.Module):
  def __init__(self,n_emb,channels,n_res):
    super().__init__()

    self.encoder = Encoder(channels,n_res)
    self.conv1 = nn.Conv2d(channels[-1],channels[-1],3,stride = 1, padding = 1)
    self.vector_quantizer = EfficientVectorQuantizer(channels[-1],n_emb)
    self.conv2 = nn.Conv2d(channels[-1],channels[-1],3,stride = 1, padding = 1)
    self.decoder = Decoder(channels[::-1],n_res)

    self.tanh = nn.Tanh()

  def forward(self,x):
    x = self.encoder(x)
    x = self.conv1(x)
    x, loss = self.vector_quantizer(x)
    x = self.conv2(x)
    x = self.decoder(x)
    x = self.tanh(x)

    return x, loss
