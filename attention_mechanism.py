import torch.nn as nn
import torch

class Self_Attn(nn.Module):
    def __init__(self, in_dim, activation):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W*H).permute(0,2,1)
        proj_key = self.key_conv(x).view(B, -1, W*H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W*H)
        out = torch.bmm(proj_value, attention.permute(0,2,1))
        out = out.view(B, C, W, H)
        return self.gamma*out + x, attention

class Local_Self_Attn(Self_Attn):
    def __init__(self, in_dim, activation):
        super().__init__(in_dim, activation)
        self.query_conv = nn.ConvTranspose2d(in_dim, in_dim//8, 1)
        self.key_conv = nn.ConvTranspose2d(in_dim, in_dim//8, 1)
        self.value_conv = nn.ConvTranspose2d(in_dim, in_dim, 1)

    def forward(self, x):
        B, C, W, H = x.size()
        x = x.view(B, 25, C, 6, 6).view(B*25, C, 6, 6)
        out, attention = super().forward(x)
        out = out.view(B, 25, C, 6, 6).view(B, 5, 5, C, 6, 6)
        out = out.permute(0,3,1,4,2,5).reshape(B, C, 30, 30)
        return out, attention