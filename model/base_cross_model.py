import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import repeat

import torch
import torch.nn as nn

class GatedSelfAttention(nn.Module):
    def __init__(self, num_q_channels, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(num_q_channels, num_heads, dropout=dropout, bias=bias)
        self.out_proj = nn.Linear(num_q_channels, num_q_channels, bias=bias)
        # 额外的门控投影：query → gate logits
        self.gate_proj = nn.Linear(num_q_channels, num_q_channels, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(num_q_channels)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        """
        x: [seq_len, batch, num_q_channels]
        """
        out_attn, _ = self.mha(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)

        out = self.out_proj(out_attn)
        gate_input = self.norm(x)               # [seq_len, batch, num_q_channels]
        gate_logits = self.gate_proj(gate_input) # [seq_len, batch, num_q_channels]
        gate = self.sigmoid(gate_logits)         # [0,1] element-wise
        gated_output = out * gate
        return gated_output


class GatedCrossAttention(nn.Module):
    def __init__(self, num_q_channels, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.mha = nn.MultiheadAttention(num_q_channels, num_heads, dropout=dropout, bias=bias)
        self.out_proj = nn.Linear(num_q_channels, num_q_channels, bias=bias)
        # 额外的门控投影：query → gate logits
        self.gate_proj = nn.Linear(num_q_channels, num_q_channels, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.norm = nn.LayerNorm(num_q_channels)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        query: [seq_len_q, batch, num_q_channels]  --> 来自解码器的输入（目标序列的表示）
        key: [seq_len_kv, batch, num_q_channels]   --> 来自编码器的输入（源序列的表示）
        value: [seq_len_kv, batch, num_q_channels] --> 来自编码器的输入（源序列的表示）
        """
        # 计算 Cross-Attention
        out_attn, _ = self.mha(query, key, value,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)

        out = self.out_proj(out_attn)

        gate_input = self.norm(query)               # [seq_len_q, batch, num_q_channels]
        gate_logits = self.gate_proj(gate_input)    # [seq_len_q, batch, num_q_channels]
        gate = self.sigmoid(gate_logits)            # [0,1] element-wise
        gated_output = out * gate                   # 逐元素乘以 gate（加权）

        return gated_output



class MultiHeadAttention(nn.Module):
    """
        Multi-Head Attention module
    """

    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)


    def forward(self, q, kv, return_atts=False):
        """
        :param q: (bs, lenq, d_model)
        :param k: (bs, lenv, d_model)
        :param v: (bs, lenv, d_model)
        :param mask:
        :return: (bs, lenq, d_model)
        """
        if not return_atts:
            summed_v = self.attention(q.permute(1, 0, 2), kv.permute(1, 0, 2), kv.permute(1, 0, 2))[0].permute(1, 0, 2)
            return q + self.dropout(summed_v)
        else:

            r = self.attention(q.permute(1, 0, 2), kv.permute(1, 0, 2), kv.permute(1, 0, 2))
            summed_v = r[0].permute(1, 0, 2)
            return q + self.dropout(summed_v), r[1]


class PositionwiseFeedForward(nn.Module):
    """ A two-feed-forward-layer module """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        return x

# RoPE
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = max_position_embeddings
        # Initialize cache
        self._update_cache(max_position_embeddings)

    def _update_cache(self, max_seq_len):
        self.max_seq_len_cached = max_seq_len
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        # [seq_len, 1, dim] for broadcasting to [seq_len, bs, dim]
        self.register_buffer("cos_cached", emb.cos()[:, None, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[:, None, :], persistent=False)

    def _apply_rotary_pos_emb(self, x, cos, sin):
        # x: [seq_len, bs, dim]
        # rotate_half
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        x_rotated = torch.cat((-x2, x1), dim=-1)
        
        return (x * cos) + (x_rotated * sin)
    
    def forward(self, x):
        # x: [seq_len, bs, dim]
        seq_len = x.shape[0]
        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len)
            
        cos = self.cos_cached[:seq_len, ...].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:seq_len, ...].to(dtype=x.dtype, device=x.device)
        
        return self._apply_rotary_pos_emb(x, cos, sin)


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class SelfAttention(nn.Module):
    """
        multi-attention + position feed forward
    """

    def __init__(self, num_heads, num_channels, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.self_att = MultiHeadAttention(num_heads, num_channels, num_channels, dropout)

    def forward(self, enc_input, return_atts=False):
        x = self.norm(enc_input)
        return self.self_att(x, x, return_atts)


class SelfAttentionLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, num_heads, num_q_channels, dropout=0.1):
        super(SelfAttentionLayer, self).__init__()
        self.self_att = SelfAttention(num_heads, num_q_channels, dropout)
        self.mlp = PositionwiseFeedForward(num_q_channels, num_q_channels, dropout)

    def forward(self, qkv, return_atts=False):
        if return_atts:
            r = self.self_att(qkv, return_atts)
            return self.mlp(r[0]), r[1]
        return self.mlp(self.self_att(qkv))


class CrossAttention(nn.Module):
    """ Compose with three layers """

    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.cross_att = MultiHeadAttention(num_heads, num_q_channels, num_kv_channels, dropout)

    def forward(self, q, kv, return_atts=False):
        q = self.q_norm(q)
        kv = self.kv_norm(kv)
        return self.cross_att(q, kv, return_atts)


class CrossAttentionLayer(nn.Module):
    """ Compose with three layers """

    def __init__(self, num_heads, num_q_channels, num_kv_channels, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.cross_att = CrossAttention(num_heads, num_q_channels, num_kv_channels, dropout)
        self.mlp = PositionwiseFeedForward(num_q_channels, num_q_channels, dropout)

    def forward(self, q, kv, return_atts=False):
        if return_atts:
            r = self.cross_att(q, kv, return_atts)
            return self.mlp(r[0]), r[1]
        return self.mlp(self.cross_att(q, kv))


class PerceiveEncoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, n_input_channels, n_latent, n_latent_channels=512, n_cross_att_heads=1,
        n_self_att_heads=8, n_self_att_layers=6, dropout=0.1, n_position=400, use_rotary_pos_emb=False):

        super().__init__()
        # TODO：尝试替换 PositionalEncoding 为 RotaryPositionalEmbedding
        if use_rotary_pos_emb:
            self.position_enc = RotaryPositionalEmbedding(n_input_channels)
        else:
            self.position_enc = PositionalEncoding(n_input_channels, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.cross_att = CrossAttentionLayer(
                    num_q_channels=n_latent_channels,
                    num_kv_channels=n_input_channels,
                    num_heads=n_cross_att_heads,
                    dropout=dropout)
        self_att = [SelfAttentionLayer(
                    num_q_channels=n_latent_channels,
                    num_heads=n_self_att_heads,
                    dropout=dropout) for _ in range(n_self_att_layers)]
        self.self_att = nn.Sequential(*self_att)
        self.latent = nn.Parameter(torch.empty(n_latent, n_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, feats_embedding):
        """
        :param feats_embedding: (bs, n_length, dim)
        :param return_atts:
        :return:
        """
        b = feats_embedding.shape[0]
        enc_output = self.dropout(self.position_enc(feats_embedding))
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        # print(x_latent.shape, enc_output.shape)
        x_latent = self.cross_att(x_latent, enc_output)
        x_latent = self.self_att(x_latent)
        return x_latent


class PerceiveDecoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(self, n_query, n_query_channels, n_latent_channels, n_cross_att_heads=1, dropout=0.1):

        super().__init__()

        self.cross_att = CrossAttentionLayer(
                    num_q_channels=n_query_channels,
                    num_kv_channels=n_latent_channels,
                    num_heads=n_cross_att_heads,
                    dropout=dropout)

        self.query_latent = nn.Parameter(torch.empty(n_query, n_query_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.query_latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, query, latent, return_atts=False):
        """
        :param query: (bs, n_q, d_q)
        :param latent: (bs, n_length, d_latent)
        :return:
        """
        b = query.shape[0]
        x_latent = repeat(self.query_latent, "... -> b ...", b=b)
        query_embedding = query + x_latent #torch.cat([query, x_latent], dim=-1)
        x_latent = self.cross_att(query_embedding, latent, return_atts)
        return x_latent


if __name__ == '__main__':
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    motion_feats = torch.randn((3, 6, 256))
    gaze_feats = torch.randn((3, 6, 256))

    gaze_encoder = PerceiveEncoder(n_input_channels=256, n_latent=10, n_latent_channels=512)
    gaze_embedding = gaze_encoder(gaze_feats)
    print(gaze_embedding.shape)

    motion_encoder = PerceiveEncoder(n_input_channels=256, n_latent=10, n_latent_channels=512)
    motion_embedding = motion_encoder(motion_feats)
    print(motion_embedding.shape)

    decoder = PerceiveDecoder(n_query=10, n_query_channels=512, n_latent_channels=512)
    print(decoder(gaze_embedding, motion_embedding).shape)