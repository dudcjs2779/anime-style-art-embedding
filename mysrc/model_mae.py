import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from timm.models.eva import Eva

def random_masking(x, mask_ratio):
    """
    x: (B, N, D) - N은 패치 수
    mask_ratio: 0.75이면 75% 마스크
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))

    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)   # noise 값이 낮은 값의 인덱스를 앞으로 정렬 => 섞는 코드를 noise로 구현했을뿐 이 ids 값으로 같은 크기의 배열을 섞을 수 있음.
    ids_restore = torch.argsort(ids_shuffle, dim=1) # ids_shuffle의 인덱스 순서를 기억해 ids_shuffle로 섞은 행렬을 ids_restore 순서대로 뽑아내면 복원이 가능 

    ids_keep = ids_shuffle[:, :len_keep]    # 마스크하지 않을 토큰 선택
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # 유지할 패치들을 가져옴

    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0  # 유지할 토큰을 0으로 설정, 1은 마스킹된 토큰 (이부분은 이미 섞인 순서로 가정, 이미 맨 앞에서부터 유지할 토큰으로 정의하고 있음.)
    mask = torch.gather(mask, dim=1, index=ids_restore) # mask를 원새 순서로 복원

    return x_masked, mask, ids_restore

class Decoder(nn.Module):
    def __init__(
            self, 
            patch_size, 
            num_patches, 
            in_chans,
            encoder_dim, 
            decoder_dim, 
            depth,
            num_heads, 
            mlp_ratio, 
            norm_layer, 
        ):
        super().__init__()
        
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=True)
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_chans, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.trunc_normal_(self.mask_token, std=.02)

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

    def forward(self, x, ids_restore):
        '''
        x: 마스킹된 토큰을 제외한 유지된 토큰
        ids_restore: 순서 복원 키
        '''
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token, x_masked(섞인 유지 토큰)을 앞에 배치하고 뒤로 mask_tokens 붙임으로써 섞인 순서를 유지
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # 순서 복원
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


class MAE(nn.Module):
    def __init__(self, encoder, decoder_embed_dim=512, decoder_depth=8, 
                 decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, 
                 mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed_dim = encoder.embed_dim  # ViT embed dim
        patch_size = encoder.patch_embed.patch_size[0]  # ViT patch size
        in_chans = encoder.patch_embed.proj.in_channels  # ViT input channels
        num_patches = encoder.patch_embed.num_patches

        # Encoder
        self.encoder = encoder

        # Decoder
        self.decoder = Decoder(
            num_patches=num_patches,
            encoder_dim=self.embed_dim,
            decoder_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            patch_size=patch_size,
            in_chans=in_chans,
        )

        # self.decoder_embed = nn.Linear(self.embed_dim, decoder_embed_dim, bias=True)
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=True)  # fixed sin-cos embedding
        # self.decoder_blocks = nn.ModuleList([
        #     Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(decoder_depth)])

        # self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3) => patch_size**2 *3: 이 사이즈는 dim=768이라는 게 아니라 패치 크기에 x 채널을 한 것.
        """
        p = self.encoder.patch_embed.patch_size[0]
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
        p = self.encoder.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def forward_encoder(self, imgs, mask_ratio):
        # 패치 임베딩
        patches = self.encoder.patch_embed(imgs)
        B, N, D = patches.shape

        x = patches + self.encoder.pos_embed[:, 1:, :]  # cls token 제외

        # 마스킹
        x_masked, mask, ids_restore = random_masking(x, mask_ratio)    # 유지된 패치(섞인 상태), 마스크 모양(복원 상태), 원래 순서 복원 인덱스

        # 인코더
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x_encoded = torch.cat([cls_tokens, x_masked], dim=1)
        for blk in self.encoder.blocks:
            x_encoded = blk(x_encoded)
        # x_encoded = self.encoder.blocks(torch.cat([cls_tokens, x_masked], dim=1))
        x_encoded = self.encoder.norm(x_encoded)

        # x_encoded = x_encoded[:, 1:, :]  # cls 제거

        return x_encoded, mask, ids_restore
    

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.decoder.forward(latent, ids_restore)

        return pred, mask