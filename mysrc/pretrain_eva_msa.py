import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import timm
from timm.models.vision_transformer import Block
import random
from tqdm import tqdm


from mysrc.utils.pos_embed import get_2d_sincos_pos_embed

def random_masking(x, mask_ratio):
    """
    x: (B, N, D) - N은 패치 수
    mask_ratio: 0.75이면 75% 마스크
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)   # noise를 정렬해 패치 순서를 랜덤하게 섞음
    ids_restore = torch.argsort(ids_shuffle, dim=1) # noise가 작은 값은 유지됨 => ids_shuffle는 노이즈의 작은 순서 => ids_restore는 ids_shuffle의 순서를 기억해 어떤 패치가 작은지 기억 => 복원 가능

    ids_keep = ids_shuffle[:, :len_keep]    # 랜덤 noise에서 크기가 작은 패치순으로 유지하는 방식
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # 마스킹된 후 유지된 패치들

    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0  # 유지된 패치 위치는 0, 마스킹된 패치 위치는 1
    mask = torch.gather(mask, dim=1, index=ids_restore) # 패치의 원래순서를 기반으로 마스크 모양을 만듦

    return x_masked, mask, ids_restore


class MAE(nn.Module):
    def __init__(self, encoder, decoder_embed_dim=512, decoder_depth=8, 
                 decoder_num_heads=8, mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 norm_pix_loss=False, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed_dim = encoder.embed_dim  # ViT embed dim
        patch_size = encoder.patch_embed.patch_size[0]  # ViT patch size
        in_chans = encoder.patch_embed.proj.in_channels  # ViT input channels
        num_patches = encoder.patch_embed.num_patches

        # Encoder
        self.encoder:timm.models.Eva = encoder
        self.patch_embed = encoder.patch_embed
        self.cls_token = encoder.cls_token
        self.pos_embed = encoder.pos_embed

        # Decoder
        self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed.requires_grad = False

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, imgs):
        # 패치 임베딩
        patches = self.patch_embed(imgs)
        B, N, D = patches.shape

        x, rot_pos_embed = self.encoder._pos_embed(patches)
        # x = patches + self.encoder.pos_embed[:, 1:, :]  # cls token 제외

        # 마스킹
        x_masked, mask, ids_restore = random_masking(x, self.mask_ratio)    # 유지된 패치, 패치가 마스킹된 모양, 원래 순서 복원 인덱스

        # 인코더
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x_encoded = self.encoder.blocks(torch.cat([cls_tokens, x_masked], dim=1), rope=rot_pos_embed)
        x_encoded = self.encoder.norm(x_encoded)

        # x_encoded = x_encoded[:, 1:, :]  # cls 제거

        return x_encoded, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

        # # 디코더 입력: 마스크 위치는 mask_token
        # mask_tokens = self.mask_token.expand(B, N - x.shape[1], -1)
        # x_full = torch.cat([x, mask_tokens], dim=1)
        # x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        # x_rec = self.decoder(x_full)  # (B, N, D)

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)

        return pred, mask


def train_one_epoch(epoch, model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(total=len(dataloader), desc=f"Training Epoch:{epoch+1}")
    for imgs, _ in dataloader:
        breakpoint()
        imgs = imgs.to(device)

        optimizer.zero_grad()
        pred, mask = model(imgs)

        target = model.patchify(imgs)

        if model.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pbar.postfix = f"Loss batch:{loss:.5f}" 
        pbar.update(1)
    pbar.close()

    return total_loss / len(dataloader)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ViT backbone
vit = timm.create_model('vit_base_patch16_224', pretrained=False)
# mae = MAE(vit, norm_pix_loss=True).to(device)
mae = MAE(vit, norm_pix_loss=True).to('cpu')
breakpoint()

# Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

# Optimizer
optimizer = torch.optim.AdamW(mae.parameters(), lr=5e-4, weight_decay=0.05)

# Train
for epoch in range(1):
    loss = train_one_epoch(epoch, mae, dataloader, optimizer, device)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")

checkpoint_dict = {
    "epoch": epoch+1,
    "name": "vit_base_patch16_224",
    "state_dict": mae.state_dict(),
    "optimizer": optimizer.state_dict(),
    'torch_rng_state': torch.random.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    'numpy_rng_state': np.random.get_state(),
    'random_state': random.getstate()
}

torch.save(
    checkpoint_dict,
    os.path.join("TestOutput/", f"epoch_{epoch}.pt"),
)