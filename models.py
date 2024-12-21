

from basemodel import *
from laplace_decoder_high_low_att import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from hgt import HGTConv_My


from my_decoder import Conv_Decoder

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self, reduction: str = 'mean') -> None:
        super(SoftTargetCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        cross_entropy = torch.sum(-target * F.log_softmax(pred, dim=-1), dim=-1)
        if self.reduction == 'mean':
            return cross_entropy.mean()
        elif self.reduction == 'sum':
            return cross_entropy.sum()
        elif self.reduction == 'none':
            return cross_entropy
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class LaplaceNLLLoss(nn.Module):

    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))

class GaussianNLLLoss(nn.Module):
    """https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html
    """
    def __init__(self,
                 eps: float = 1e-6,
                 reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        # print("scale",scale.shape,"loc",loc.shape)
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = 0.5*(torch.log(scale**2) + torch.abs(target - loc)**2 / scale**2)
        # print("nll", nll.shape)
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        elif self.reduction == 'none':
            return nll
        else:
            raise ValueError('{} is not a valid value for reduction'.format(self.reduction))


class EPHGT(nn.Module):
    def __init__(self, args):
        super(EPHGT, self).__init__()
        self.args = args
        self.Temperal_Encoder = Temperal_Encoder(self.args)
        self.Laplacian_Decoder = GRUDecoder(self.args)
        
        
        if self.args.ifGaussian:
            self.reg_loss = GaussianNLLLoss(reduction='mean')
        else:
            self.reg_loss = LaplaceNLLLoss(reduction='mean')
        self.cls_loss = SoftTargetCrossEntropyLoss(reduction='mean')

        self.mlp1 = nn.Sequential(
            nn.Linear(args.hidden_size, 2*args.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(2*args.hidden_size, 2*args.hidden_size),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(args.hidden_size, 2*args.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(2*args.hidden_size, 2*args.hidden_size),
        )


        self.n_hgt = 2
        self.hgt = nn.ModuleList()
        for t in range(self.n_hgt):
            self.hgt.append(HGTConv_My(args.hidden_size, args.hidden_size, num_types=6, num_relations=1, n_heads=8))

        self.to12 = Conv_Decoder(args)

    def forward(self, inputs, edge_pair, epoch, iftest=False):
        
        batch_abs_gt, batch_norm_gt, nei_list_batch, nei_num_batch, batch_split, max_values = inputs # #[H, N, 2], [H, N, 2], [B, H, N, N], [N, H], [B, 2]
        device = torch.device(batch_abs_gt.device)


        # self.batch_norm_gt = batch_norm_gt
        batch_class = torch.zeros_like(batch_abs_gt[0,:,-1]).long()
        # batch_class[batch_class==2] = 1
        # batch_class[batch_class>0] = 1
        batch_abs_gt = batch_abs_gt[:8,:,:2]
        self.batch_norm_gt = batch_norm_gt
        if self.args.input_offset:
            train_x = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
            zeros = torch.zeros(1, train_x.size(1), 2, device=device)
            train_x = torch.cat([zeros, train_x], dim=0)
        elif self.args.input_mix:
            offset = batch_norm_gt[1:self.args.obs_length, :, :] - batch_norm_gt[:self.args.obs_length-1, :, :] #[H, N, 2]
            position = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
            pad_offset = torch.zeros_like(position, device=device)
            pad_offset[1:, :, :] = offset
            train_x = torch.cat((position, pad_offset), dim=2)
        elif self.args.input_position:
            train_x = batch_norm_gt[:self.args.obs_length, :, :] #[H, N, 2]
        train_x = train_x.permute(1, 2, 0) #[N, 2, H]
        train_y = batch_norm_gt[self.args.obs_length:, :, :].permute(1, 2, 0) #[N, 2, H]
        self.pre_obs=batch_norm_gt[1:self.args.obs_length]
        
        x_encoded, hidden_state_unsplited, cn = self.Temperal_Encoder.forward(train_x)  #[N, H, 2], [N, D], [N, D]
        # to12
        x_encoded = self.to12(x_encoded)
        
        
        
        
        batch_abs_gt = batch_abs_gt[7, :, :] # 最后一个观测时刻的绝对坐标 [N, 2
        #print(batch_abs_gt.shape, max_values.shape)
        
        batch_norm_abs_gt = (batch_abs_gt / max_values[0]).float()

        hidden_state_global = hidden_state_unsplited
        cn_global = cn
        priority_full = torch.zeros(hidden_state_unsplited.size(0), device=device)

        for left, right in batch_split:
            now_pair = edge_pair[(left.item(), right.item())][0].to(device).long()
            if len(now_pair) != 0:
                edge_pair_now = now_pair.transpose(0, 1)
                edge_type_now = torch.zeros(edge_pair_now.size(-1), device=device) # 同一类

                hidden_now = hidden_state_unsplited[left: right].view(-1, self.args.hidden_size)
                cn_now = cn[left: right]
                x_norm_abs_now = batch_norm_abs_gt[left: right] # 绝对坐标

                node_type = batch_class[left: right]

                for t in range(self.n_hgt):
                    hgt_out, priority = self.hgt[t](hidden_now, x_norm_abs_now, node_type, edge_pair_now, edge_type_now)

                    cn_now = hgt_out + cn_now
                    hidden_now = hidden_now + F.tanh(cn_now)

                priority_full[left: right] = priority
                hidden_state_global[left: right] = hidden_now
                cn_global[left: right] = cn_now

            else: # 没有边
                pass
                    
        train_y_gt = train_y.permute(0, 2, 1)
        

        mdn_out, (loc1, scale1) = self.Laplacian_Decoder.forward(x_encoded, hidden_state_global, cn_global, priority_full, batch_split)
        


        EPHGT_loss, full_pre_tra = self.mdn_loss(train_y_gt, mdn_out, 1, True)  #[K, H, N, 2]
        EPHGT_loss1, full_pre_tra1 = self.mdn_loss(train_y_gt, (loc1, scale1), 1, False)  #[K, H, N, 2]
        return (EPHGT_loss, EPHGT_loss1), full_pre_tra

    def mdn_loss(self, y, y_prime, goal_gt, ifpi):
        batch_size=y.shape[0]
        # y = y.permute(1, 0, 2)  #[N, H, 2]
        # [F, N, H, 2], [F, N, H, 2], [N, F]
        if ifpi:
            out_mu, out_sigma, out_pi = y_prime
        else:
            out_mu, out_sigma = y_prime
        y_hat = torch.cat((out_mu, out_sigma), dim=-1)
        reg_loss, cls_loss = 0, 0
        full_pre_tra = []
        l2_norm = (torch.norm(out_mu - y, p=2, dim=-1) ).sum(dim=-1)   # [F, N]
        best_mode = l2_norm.argmin(dim=0) # N个中每一个20中的最小的坐标
        y_hat_best = y_hat[best_mode, torch.arange(batch_size)]
        reg_loss += self.reg_loss(y_hat_best, y)
        soft_target = F.softmax(-l2_norm / self.args.pred_length, dim=0).t().detach() # [N, F]
        if ifpi:
            cls_loss += self.cls_loss(out_pi, soft_target)
            loss = reg_loss + cls_loss
        else:
            loss = reg_loss
        #best ADE
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        # best FDE
        l2_norm_FDE = (torch.norm(out_mu[:,:,-1,:] - y[:,-1,:], p=2, dim=-1) )  # [F, N]
        best_mode = l2_norm_FDE.argmin(dim=0)
        sample_k = out_mu[best_mode, torch.arange(batch_size)].permute(1, 0, 2)  #[H, N, 2]
        full_pre_tra.append(torch.cat((self.pre_obs,sample_k), axis=0))
        return loss, full_pre_tra
