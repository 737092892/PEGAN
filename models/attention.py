import torch.nn as nn
import torch
import torch.nn.functional as F


class Self_Attn(nn.Module):
    """ Self attention Layer with Sparse Attention Matrix """
    def __init__(self, in_dim, activation, window_size=3):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.window_size = window_size  # Size of the local attention window

        # Convolution layers for query, key, and value
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature maps (B x C x W x H)
        returns :
            out : self attention value + input feature
            attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        N = width * height

        # Project queries, keys, and values
        proj_query = self.query_conv(x).view(m_batchsize, -1, N).permute(0, 2, 1)  # B x N x C'
        proj_key = self.key_conv(x).view(m_batchsize, -1, N)  # B x C' x N
        proj_value = self.value_conv(x).view(m_batchsize, -1, N)  # B x C x N

        # Initialize sparse attention matrix
        attention = torch.zeros(m_batchsize, N, N, device=x.device)  # B x N x N

        # Compute sparse attention scores within a local window
        for i in range(N):
            # Get the 2D position (row, col) of the current pixel
            row, col = i // height, i % height

            # Define the local window boundaries
            row_start = max(0, row - self.window_size // 2)
            row_end = min(width, row + self.window_size // 2 + 1)
            col_start = max(0, col - self.window_size // 2)
            col_end = min(height, col + self.window_size // 2 + 1)

            # Flatten the local window indices
            local_indices = []
            for r in range(row_start, row_end):
                for c in range(col_start, col_end):
                    local_indices.append(r * height + c)

            # Compute attention scores only for the local window
            local_query = proj_query[:, i:i+1, :]  # B x 1 x C'
            local_key = proj_key[:, :, local_indices]  # B x C' x K
            local_energy = torch.bmm(local_query, local_key)  # B x 1 x K
            local_attention = self.softmax(local_energy)  # B x 1 x K

            # Fill the sparse attention matrix
            attention[:, i, local_indices] = local_attention.squeeze(1)  # B x K

        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x N
        out = out.view(m_batchsize, C, width, height)  # Reshape to original dimensions

        # Add residual connection
        out = self.gamma * out + x
        return out, attention
