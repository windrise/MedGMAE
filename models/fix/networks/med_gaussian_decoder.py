import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

class MedGaussianViTDecoder(nn.Module):
    """
    Vision Transformer decoder for Medical GMAE (Gaussian Masked Autoencoder) that outputs Gaussian parameters
    for medical imaging data
    """
    def __init__(self, num_gaussians=512, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None, patch_size=None):  # Added patch_size parameter but we don't use it
        super().__init__()
        self.num_gaussians = num_gaussians
        self.embed_dim = embed_dim
        self.num_tokens = 1  # don't consider distillation here
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        
        # We don't use patch_size but we accept it to maintain API compatibility with MAE

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Different heads for different Gaussian parameters based on your medical-specific implementation
        # Each Gaussian is parameterized by:
        # - 3D position (xyz)
        # - 3D scale (scale)
        # - Rotation quaternion (rotation)
        # - Density (opacity)
        
        # Position head: 3 values (x, y, z)
        self.position_head = nn.Linear(embed_dim, 3)
        
        # Scale head: 3 values (sx, sy, sz)
        self.scale_head = nn.Linear(embed_dim, 3)
        
        # Rotation head: 4 values (quaternion)
        self.rotation_head = nn.Linear(embed_dim, 4)
        
        # Density head: 1 value
        self.density_head = nn.Linear(embed_dim, 1)
        

        self._custom_init_heads()
        

        self.apply(self._init_weights)

    def _custom_init_heads(self):

        nn.init.xavier_uniform_(self.position_head.weight)
        nn.init.zeros_(self.position_head.bias)


        nn.init.normal_(self.scale_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.scale_head.bias, -1.386)  

 
        nn.init.xavier_uniform_(self.rotation_head.weight)
        nn.init.zeros_(self.rotation_head.bias)


        nn.init.xavier_uniform_(self.density_head.weight)
        nn.init.constant_(self.density_head.bias, -0.405) 

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):

            if m not in [self.position_head, self.scale_head, self.rotation_head, self.density_head]:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        """
        Forward pass through the Gaussian decoder
        
        Args:
            x: [B, N, C] - Input tokens where B is batch size, N is number of tokens, C is embedding dimension
                          - The first token is the class token, the rest are patch tokens or mask tokens
                          
        Returns:
            Gaussian parameters that match the medical imaging implementation:
                positions: [B, K, 3] - 3D positions (xyz)
                scales: [B, K, 3] - 3D scales
                rotations: [B, K, 4] - Rotation quaternions
                densities: [B, K, 1] - Density values
        """
        # print("x shape: ", x.shape)  # torch.Size([8, 769, 528])
        x = self.forward_features(x)
        # print("x shape: ", x.shape)  # torch.Size([8, 769, 528])
        # Discard the class token and use the query tokens (positions 1:num_gaussians+1)
        # to generate Gaussian parameters
        gaussian_tokens = x[:, 1:self.num_gaussians+1]
        

        positions = torch.sigmoid(self.position_head(gaussian_tokens))
        

        scales = torch.sigmoid(self.scale_head(gaussian_tokens))


        rotations = self.rotation_head(gaussian_tokens)

        rotations = nn.functional.normalize(rotations, dim=-1)
        

        densities = torch.sigmoid(self.density_head(gaussian_tokens))

        return positions, scales, rotations, densities

def med_gaussian_decoder_base(**kwargs):
    model = MedGaussianViTDecoder(
        embed_dim=384,
        depth=8,
        num_heads=12,
        **kwargs)
    return model

def med_gaussian_decoder_large(**kwargs):
    model = MedGaussianViTDecoder(
        embed_dim=528,
        depth=8,
        num_heads=16,
        **kwargs)
    return model


