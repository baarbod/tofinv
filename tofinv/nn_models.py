import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, dim, kernel_size=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding='same'),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x)) # Residual connection

class TOFinverse(nn.Module):
    def __init__(self, nflow_in, nfeature_out, context_dim=32, hidden_dim=128, num_blocks=4):
        """
        Deep dual-input network for TOF inversion.
        num_blocks: Increase this to make the network deeper (e.g., 4, 6, 8).
        """
        super().__init__()
        
        # 1. Flow Feature Extractor
        self.flow_in = nn.Sequential(
            nn.Conv1d(nflow_in, hidden_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        # 2. Area Context Extractor
        self.area_in = nn.Sequential(
            nn.Conv1d(1, context_dim, kernel_size=5, padding='same'),
            nn.BatchNorm1d(context_dim),
            nn.GELU()
        )
        
        # 3. Deep Processing (Combined Features)
        combined_dim = hidden_dim + context_dim
        
        # Stack residual blocks for depth without vanishing gradients
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock1D(combined_dim))
        self.deep_processor = nn.Sequential(*blocks)
        
        # 4. Final Projection to Velocity
        self.out_conv = nn.Sequential(
            nn.Conv1d(combined_dim, hidden_dim // 2, kernel_size=3, padding='same'),
            nn.GELU(),
            nn.Conv1d(hidden_dim // 2, nfeature_out, kernel_size=3, padding='same')
        )

    def forward(self, flow_x, area_x):
        # Extract initial features
        f_feat = self.flow_in(flow_x)
        a_feat = self.area_in(area_x)
        
        # Concatenate along the channel dimension
        combined = torch.cat([f_feat, a_feat], dim=1)
        
        # Pass through deep residual blocks
        deep_feat = self.deep_processor(combined)
        
        # Predict output
        return self.out_conv(deep_feat)

