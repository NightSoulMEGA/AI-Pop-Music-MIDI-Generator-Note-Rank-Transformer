import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb    #dimension of positional embedding

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):   #pos_seq stands for the sequence's position
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):    #FF=Feed Forward
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = self.CoreNet(inp)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head    #number of attention heads
        self.d_model = d_model  #dimension of hidden layers
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)  #matrix to calculate query, key and value

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)    #matrix to concatenate the results of attention heads and then transform them into the model dimension

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        mask = torch.ones((h, w)).byte()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError

class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False) #用于计算相对位置编码变换的参数矩阵

    def forward(self, w, condition, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        #将歌词嵌入拼接到memory之前
        if mems is not None:
            cat = torch.cat([condition, mems, w], dim=0)   #将condition，memory，word_emb连接在一起
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head

        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias
        AC = rw_head_q.permute(1, 2, 0, 3) @ w_head_k.permute(1, 2, 3, 0)

        rr_head_q = w_head_q + r_r_bias
        BD = rr_head_q.permute(1, 2, 0, 3) @ r_head_k.permute(1, 2, 0)
        BD = F.pad(BD, [1, 0]).view(BD.size(0), BD.size(
            1), BD.size(3) + 1, BD.size(2))[:, :, 1:].view_as(BD)

        # [bsz x n_head x qlen x klen]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask, -float('inf')).type_as(attn_score)
            elif attn_mask.dim() == 3:
                attn_score = attn_score.float().masked_fill(
                    attn_mask.permute(2, 0, 1)[:, None, :, :], -float('inf')).type_as(attn_score)

        # [bsz x n_head x qlen x klen]
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropatt(attn_prob)

        #### compute attention vector
        attn_vec = attn_prob @ w_head_v.permute(1, 2, 0, 3)
        attn_vec = attn_vec.permute(2, 0, 1, 3)

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            ##### residual connection
            output = w + attn_out
        else:
            ##### residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, 
                                     pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, condition, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, condition, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)



class MemTransformerLM(nn.Module):
    def __init__(self, modelConfig,is_training=True):
        super(MemTransformerLM, self).__init__()

        self.n_token = modelConfig['n_token']
        self.n_layer= modelConfig['n_layer']
        self.n_head= modelConfig['n_head']
        self.d_model = modelConfig['d_model']
        self.d_embed = self.d_model if modelConfig['d_embed'] is None else modelConfig['d_embed']
        self.d_head = self.d_model // self.n_head
        self.d_inner= modelConfig['d_inner']

        self.mem_len = modelConfig['mem_len']
        self.tgt_len = modelConfig['tgt_len']
        self.ext_len = modelConfig['ext_len']

        self.dropout= modelConfig['dropout']
        self.dropatt = modelConfig['dropatt']

        self.clamp_len = modelConfig['clamp_len']
        self.div_val = modelConfig['div_val']

        #choice
        self.pre_lnorm = modelConfig['pre_lnorm']
        self.same_length = modelConfig['same_length']
        self.is_training = is_training

        #building layers
        self.drop = nn.Dropout(self.dropout)    #dropout layer
        self.word_emb = Embeddings(self.n_token, self.d_model)  #embedding layer

        self.layers = nn.ModuleList()
        for i in range(self.n_layer):   #n*decoder
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    self.n_head, self.d_model, self.d_head, self.d_inner, self.dropout,
                    tgt_len=self.tgt_len, ext_len=self.ext_len, mem_len=self.mem_len,
                    dropatt=self.dropatt, pre_lnorm=self.pre_lnorm)
            )

        # output layer
        self.linear_proj = nn.Linear(self.d_model, self.n_token)    #linear layer

        # loss
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self._create_params()

    def compute_loss(self, predict, target, loss_mask=None):
        '''
        predict, target,
        input:  (N, C, ...)
        target: (N, ...)
        '''
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer+1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)
            return mems
        else:
            return None

    def _update_mems(self, hids, mems, mlen, qlen):
        
        if mems is None: return None
        # mems is not None
        # assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)

            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems



    def _forward(self, dec_inp, condition, mems=None):
        '''
        output of _forward: step x batch x n_feat
        predict = self.linear_proj(hidden)
        '''

        qlen, bsz = dec_inp.size()  #1或512,8
        clen = 128
        mlen = mems[0].size(0) if mems is not None else 0   #0或512
        klen = mlen + qlen + clen  #640,1152,129,641

        word_emb = self.word_emb(dec_inp)

        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen-clen)
            mask_len = klen - clen - self.mem_len

            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            #修改以适应添加condition后的嵌入长度
            dec_attn_mask = torch.cat([word_emb.new_zeros(qlen,clen),(torch.triu(all_ones, 1+mlen)
                    + torch.tril(all_ones, -mask_shift_len))],1).bool()[:, :, None]
        else:
            dec_attn_mask = torch.triu(
                word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]


        hids = []
        pos_seq = torch.arange(klen-1, -1, -1.0, device=word_emb.device, 
                                dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        core_out = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)
        hids.append(core_out)

        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            
            core_out = layer(core_out, condition, pos_emb, self.r_w_bias,
                    self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)

        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems

    def generate(self, data, condition, *mems):
        if not mems: mems = self.init_mems()
        #print(data.shape)   #[1,1]
        #print(condition.shape)  #[128,512]
        condition=condition.reshape(128,1,512)
        hidden, new_mems = self._forward(data, condition, mems=mems)
        pred_hid = hidden[-1:]
        predict = self.linear_proj(pred_hid)
        return predict, new_mems

    def forward(self, data, target, mask, condition, *mems):   #训练
        if not mems: mems = self.init_mems()

        # print(data.shape)   #[512,8]
        # print(target.shape) #[512,8]
        # print(mask.shape)   #[8,512]
        # print(condition.shape)  #[8,128,512]

        condition = condition.permute(1,0,2)    #[128,8,512]

        tgt_len = target.size(0)
        hidden, new_mems = self._forward(data, condition, mems=mems)

        pred_hid = hidden[-tgt_len:]
        predict = self.linear_proj(pred_hid)

        predict = predict.permute(1, 2, 0)
        target = target.permute(1, 0)

        loss = self.compute_loss(predict, target, mask)

        if new_mems is None:
            return [loss]
        else:
            return [loss] + new_mems
