import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import timm  # ViT 백본 사용
import random
from tqdm import tqdm

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
    def __init__(self, encoder, decoder_dim=512, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.embed_dim = encoder.embed_dim  # ViT embed dim

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, self.embed_dim)  # 임베딩 복원
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

    def forward_encoder(self, imgs):
        # 패치 임베딩
        patches = self.patch_embed(imgs)
        B, N, D = patches.shape

        x = patches + self.encoder.pos_embed[:, 1:, :]  # cls token 제외

        # 마스킹
        x_masked, mask, ids_restore = random_masking(x, self.mask_ratio)    # 유지된 패치, 패치가 마스킹된 모양, 원래 순서 복원 인덱스

        # 인코더
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x_encoded = self.encoder.blocks(torch.cat([cls_tokens, x_masked], dim=1))
        x_encoded = self.encoder.norm(x_encoded)

        # x_encoded = x_encoded[:, 1:, :]  # cls 제거

        return x_encoded, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):

        # 디코더 입력: 마스크 위치는 mask_token
        mask_tokens = self.mask_token.expand(B, N - x.shape[1], -1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))

        x_rec = self.decoder(x_full)  # (B, N, D)

    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs, self.mask_ratio)



    

def train_one_epoch(epoch, model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    pbar = tqdm(total=len(dataloader), desc=f"Training Epoch:{epoch+1}")
    for imgs, _ in dataloader:
        imgs = imgs.to(device)

        optimizer.zero_grad()
        pred, target, mask = model(imgs)

        # patch-level MSE on masked positions only
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()

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
mae = MAE(vit).to(device)

# Data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Optimizer
optimizer = torch.optim.AdamW(mae.parameters(), lr=1e-4)

# Train
for epoch in range(10):
    loss = train_one_epoch(epoch, mae, dataloader, optimizer, device)
    print(f"Epoch {epoch}, Loss: {loss:.4f}")
