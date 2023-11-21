import torch
import math
from torch import nn
from modules import ConvSC, Inception, ConvNeXt_block, Learnable_Filter, Attention

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Time_MLP(nn.Module):
    def __init__(self, dim):
        super(Time_MLP, self).__init__()
        self.sinusoidaposemb = SinusoidalPosEmb(dim)
        self.linear1 = nn.Linear(dim, dim*4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(dim*4, dim)

    def forward(self, x):
        x = self.sinusoidaposemb(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

def stride_generator(N, reverse=False):
    strides = [1, 2]*10
    if reverse: return list(reversed(strides[:N]))
    else: return strides[:N]

class Encoder(nn.Module):
    def __init__(self,C_in, C_hid, N_S):
        super(Encoder,self).__init__()
        strides = stride_generator(N_S)
        self.enc = nn.Sequential(
            ConvSC(C_in, C_hid, stride=strides[0]),
            *[ConvSC(C_hid, C_hid, stride=s) for s in strides[1:]]
        )

    def forward(self,x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1,len(self.enc)):
            latent = self.enc[i](latent)
        return latent,enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, N_S, shape_in):
        super(Decoder,self).__init__()
        self.shape_in = shape_in
        T, C, H, W = self.shape_in
        strides = stride_generator(N_S, reverse=True)
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, stride=s, transpose=True) for s in strides[:-1]],
            ConvSC(2*C_hid, C_hid, stride=strides[-1], transpose=True)
        )
        self.readout = nn.Conv2d(C_hid * T, C_hid, 1)

    def forward(self, hid, enc1=None):
        for i in range(0,len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](torch.cat([hid, enc1], dim=1))
        ys = Y.shape
        T, _, H, W = self.shape_in
        Y = Y.reshape(int(ys[0]/T), int(ys[1]*T), H, W)
        Y = self.readout(Y)
        return Y

class Mid_Xnet(nn.Module):
    def __init__(self, channel_in, channel_hid, N_T, incep_ker = [3,5,7,11], groups=8):
        super(Mid_Xnet, self).__init__()

        self.N_T = N_T
        enc_layers = [ConvNeXt_block(dim=channel_in)]
        for i in range(1, N_T-1):
            enc_layers.append(ConvNeXt_block(dim=channel_in))
        enc_layers.append(ConvNeXt_block(dim=channel_in))

        dec_layers = [ConvNeXt_block(dim=channel_in)]
        for i in range(1, N_T-1):
            dec_layers.append(ConvNeXt_block(dim=channel_in))
        dec_layers.append(ConvNeXt_block(dim=channel_in))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x, t):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N_T):
            z = self.enc[i](z, t)
            if i < self.N_T - 1:
                skips.append(z)

        # decoder
        z = self.dec[0](z, t)
        for i in range(1, self.N_T):
            z = self.dec[i](z, t)

        y = z.reshape(B, T, C, H, W)
        return y


class MISO(nn.Module):
    def __init__(self, shape_in, hid_S=64, hid_T=256, N_S=4, N_T=8, incep_ker=[3,5,7,11], groups=8):
        super(MISO, self).__init__()
        T, C, H, W = shape_in
        self.time_mlp = Time_MLP(dim=hid_S)
        self.enc = Encoder(C, hid_S, N_S)
        self.hid = Mid_Xnet(T*hid_S, hid_T, N_T, incep_ker, groups)
        self.dec = Decoder(hid_S, N_S, shape_in)
        self.attn = Attention(hid_S)
        self.readout = nn.Conv2d(hid_S, C, 1)

    def forward(self, x_raw, t):
        B, T, C, H, W = x_raw.shape
        x = x_raw.view(B*T, C, H, W)
        time_emb = self.time_mlp(t)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape

        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z, time_emb)
        hid = hid.reshape(B*T, C_, H_, W_)

        Y = self.dec(hid, skip)
        Y = self.attn(Y)
        Y = self.readout(Y)
        return Y
