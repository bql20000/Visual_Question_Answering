# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.model.net_utils import FC, MLP, LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math


# ------------------------------
# ---- Multi-Head Attention ----
# ------------------------------

class MHAtt(nn.Module):
    def __init__(self, __C):
        super(MHAtt, self).__init__()
        self.__C = __C

        self.linear_v = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_k = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_q = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)
        self.linear_merge = nn.Linear(__C.HIDDEN_SIZE, __C.HIDDEN_SIZE)

        self.dropout = nn.Dropout(__C.DROPOUT_R)

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.__C.MULTI_HEAD,
            self.__C.HIDDEN_SIZE_HEAD
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.__C.HIDDEN_SIZE
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


# ---------------------------
# ---- Feed Forward Nets ----
# ---------------------------

class FFN(nn.Module):
    def __init__(self, __C):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FF_SIZE,
            out_size=__C.HIDDEN_SIZE,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)


# ------------------------
# ---- Self Attention ----
# ------------------------

class SA(nn.Module):
    def __init__(self, __C):
        super(SA, self).__init__()

        self.mhatt = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, x_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.ffn(x)
        ))

        return x


# -------------------------------
# ---- Self Guided Attention ----
# -------------------------------

class SGA(nn.Module):
    def __init__(self, __C):
        super(SGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(y, y, x, y_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.ffn(x)
        ))

        return x


# --------------------------------------
# ---- Self Guided Guided Attention ----
# --------------------------------------

class SGGA(nn.Module):
    def __init__(self, __C):
        super(SGGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.mhatt3 = MHAtt(__C)
        self.ffn = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout4 = nn.Dropout(__C.DROPOUT_R)
        self.norm4 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, z, x_mask, y_mask, z_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        x = self.norm2(x + self.dropout2(
            self.mhatt2(z, z, x, z_mask)
        ))

        x = self.norm3(x + self.dropout3(
            self.mhatt3(y, y, x, y_mask)
        ))

        x = self.norm4(x + self.dropout4(
            self.ffn(x)
        ))

        return x


# -------------------------------------------
# ---- Self Guided Self Guided Attention ----
# -------------------------------------------

class SGSGA(nn.Module):
    def __init__(self, __C):
        super(SGSGA, self).__init__()

        self.mhatt1 = MHAtt(__C)
        self.mhatt2 = MHAtt(__C)
        self.mhatt3 = MHAtt(__C)
        self.mhatt4 = MHAtt(__C)
        self.ffn1 = FFN(__C)
        self.ffn2 = FFN(__C)

        self.dropout1 = nn.Dropout(__C.DROPOUT_R)
        self.norm1 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(__C.DROPOUT_R)
        self.norm2 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(__C.DROPOUT_R)
        self.norm3 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout4 = nn.Dropout(__C.DROPOUT_R)
        self.norm4 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout5 = nn.Dropout(__C.DROPOUT_R)
        self.norm5 = LayerNorm(__C.HIDDEN_SIZE)

        self.dropout6 = nn.Dropout(__C.DROPOUT_R)
        self.norm6 = LayerNorm(__C.HIDDEN_SIZE)

    def forward(self, x, y, x_mask, y_mask):
        x = self.norm1(x + self.dropout1(
            self.mhatt1(x, x, x, x_mask)
        ))

        y = self.norm2(y + self.dropout2(
            self.mhatt2(y, y, y, y_mask)
        ))

        x_ = self.norm3(x + self.dropout3(
            self.mhatt3(y, y, x, y_mask)
        ))

        y_ = self.norm4(y + self.dropout4(
            self.mhatt4(x, x, y, x_mask)
        ))

        x_ = self.norm5(x_ + self.dropout5(
            self.ffn1(x_)
        ))

        y_ = self.norm6(y_ + self.dropout6(
            self.ffn2(y_)
        ))
        
        return x_, y_


# ------------------------------------------------
# ---- MAC Layers Cascaded by Encoder-Decoder ----
# ------------------------------------------------

class MCA_ED(nn.Module):
    def __init__(self, __C):
        super(MCA_ED, self).__init__()

        self.enc_list = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGA(__C) for _ in range(__C.LAYER)])

    def forward(self, x, y, x_mask, y_mask):
        # Get hidden vector
        for enc in self.enc_list:
            y = enc(y, y_mask)

        for dec in self.dec_list:
            x = dec(x, y, x_mask, y_mask)

        return x, y


# ----------------------------------------------
# ---- MCA Layers with Gradual Guiding Tags ----
# ----------------------------------------------

class MCA_Gradual(nn.Module):
    def __init__(self, __C):
        super(MCA_Gradual, self).__init__()

        self.enc_list1 = nn.ModuleList([SA(__C) for _ in range(__C.G_LAYER)])
        self.enc_list2 = nn.ModuleList([SA(__C) for _ in range(__C.G_LAYER)])
        # self.enc_list3 = nn.ModuleList([SA(__C) for _ in range(__C.LAYER)])
        self.dec_list = nn.ModuleList([SGGA(__C) for _ in range(__C.G_LAYER)])

    def forward(self, x, y, z, x_mask, y_mask, z_mask):
        # Get hidden vectors
        for enc1 in self.enc_list1:
            y = enc1(y, y_mask)
        for enc2 in self.enc_list2:
            z = enc2(z, z_mask)

        # for enc3 in self.enc_list3:
        #     x = enc3(x, x_mask)

        for dec in self.dec_list:
            x = dec(x, y, z, x_mask, y_mask, z_mask)

        return x, y


# ------------------------------------------------
# ---- MCA Layers with Cross Guided Attention ----
# ------------------------------------------------

class MCA_Cross(nn.Module):
    def __init__(self, __C):
        super(MCA_Cross, self).__init__()

        self.enc_list1 = nn.ModuleList([SA(__C) for _ in range(__C.I_LAYER)])
        self.enc_list2 = nn.ModuleList([SA(__C) for _ in range(__C.Q_LAYER)])
        self.dec_list1 = nn.ModuleList([SGA(__C) for _ in range(__C.I_LAYER)])
        self.dec_list2 = nn.ModuleList([SGA(__C) for _ in range(__C.Q_LAYER)])
        self.cross_list = nn.ModuleList([SGSGA(__C) for _ in range(__C.I_LAYER)])

    def forward(self, x, y, z, x_mask, y_mask, z_mask):
        z_1 = z_2 = z
        # Get hidden vectors

        #Tag_Imgae
        for enc1 in self.enc_list1:
            z_1 = enc1(z_1, z_mask)
        #Tag_Question
        for enc2 in self.enc_list2:
            z_2 = enc2(z_2, z_mask)

        #Image
        for dec1 in self.dec_list1:
            x = dec1(x, z_1, x_mask, z_mask)

        #Question
        for dec2 in self.dec_list2:
            y = dec2(y, z_2, y_mask, z_mask)
        
        for cross in self.cross_list:
            x, y = cross(x, y, x_mask, y_mask)
        
        return x, y
