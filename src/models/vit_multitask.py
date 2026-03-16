import torch
import torch.nn as nn
import timm

from src.config import VIT_MODEL_NAME, NUM_LABELS, GRID_SIZE


class ViTMultiTask(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, grid_size=GRID_SIZE,
                 model_name=VIT_MODEL_NAME, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        d = self.vit.embed_dim
        self.grid_size = grid_size

        # classification head, multi-label
        self.cls_head = nn.Linear(d, num_labels)

        # saliency head, one value per patch token
        self.sal_head = nn.Linear(d, 1)

    def forward(self, x):
        feats = self.vit.forward_features(x)  # (b, 1+num_patches, d)

        cls_tok = feats[:, 0, :]
        patch_toks = feats[:, 1:, :]

        logits = self.cls_head(cls_tok)  # (b, num_labels)

        s = self.sal_head(patch_toks).squeeze(-1)  # (b, num_patches)
        # softmax so output is a valid probability distribution over patches
        s = torch.softmax(s, dim=-1)
        h = self.grid_size
        sal_grid = s.view(s.size(0), 1, h, h)  # (b, 1, 14, 14)

        return logits, sal_grid

    def get_attention_weights(self, x):
        # extract attention weights from each block manually
        # returns list of (b, heads, tokens, tokens)
        B = x.shape[0]
        x_internal = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x_internal = torch.cat((cls_token, x_internal), dim=1)
        x_internal = x_internal + self.vit.pos_embed
        x_internal = self.vit.pos_drop(x_internal)

        attn_weights = []
        for block in self.vit.blocks:
            # manually compute attention
            B_n, N, C = x_internal.shape
            qkv = block.attn.qkv(x_internal).reshape(B_n, N, 3, block.attn.num_heads, C // block.attn.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn = attn.softmax(dim=-1)
            attn_weights.append(attn)  # (b, heads, tokens, tokens)

            # continue forward through the block
            x_internal = block(x_internal)

        return attn_weights


def build_model(mode="multitask", num_labels=NUM_LABELS, pretrained=True):
    # mode: "multitask", "cls_only", "sal_only"
    model = ViTMultiTask(num_labels=num_labels, pretrained=pretrained)
    return model, mode
