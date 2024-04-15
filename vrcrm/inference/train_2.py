import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm

from ..gumbel.gumbel_multilabel import *
from ..models import Policy, T


def train2(max_epoch: int, bandit_train_loader: DataLoader,
        fgan_loader: DataLoader, hnet: Policy, Dnet_xy: T,
        steps_fgan: int, device,
        is_gumbel_hard: bool = False,
        opts=None):
    is_cuda = device.type == "cuda"
    opt_h, opt_h2, opt_d = opts

    for epoch in tqdm(range(max_epoch)):
        for ele in bandit_train_loader:
            X, s_labels, s_log_prop, s_loss, y = ele

            X = X.float().to(device)
            s_labels = s_labels.to(device)
            s_log_prop = s_log_prop.to(device)
            s_loss = s_loss.to(device)
            y = y.to(torch.int64).to(device)
            idx = y

            opt_h.zero_grad()
            hnet.train()
            prob = hnet(X)


            # compute expected loss
            h = (1-s_labels) * (1-prob) + s_labels * prob
            stacked = torch.stack((prob, 1-prob)) #[2, bs, nl], with idx [bs, nl]
            # idx[i,j]=1, select stack[0,i,j]
            idx = idx.view((1, idx.size(0), idx.size(1)))
            h = torch.gather(stacked,dim=0,index=1-idx).squeeze() # [bs, nl]

            prob_per_instance = torch.sum(h, dim=1)
            log_prob_per_instance = torch.log(prob_per_instance+1e-9)
            log_IS = log_prob_per_instance - s_log_prop
            loss = torch.mean(s_loss * torch.exp(log_IS))

            # update paras of hnet
            loss.backward()
            opt_h.step()
            opt_h.zero_grad()

            hnet.eval()
            hnet.train()

            algorithm_1(steps_fgan=steps_fgan, fgan_loader=fgan_loader, device=device, hnet=hnet, is_gumbel_hard=is_gumbel_hard, is_cuda=is_cuda, Dnet_xy=Dnet_xy, opts=opts)

def algorithm_1(steps_fgan, fgan_loader, device, hnet, is_gumbel_hard, is_cuda, Dnet_xy, opts):
    opt_h, opt_h2, opt_d = opts

    # f-gan training
    # hnet.eval()
    fgan_counter = 0
    #for i in range(3)
    for i in range(steps_fgan):
        for ele in fgan_loader:
            fgan_counter += 1

            # dataset convertion
            X, s_labels, s_log_prop, s_loss, y = ele
            X = X.to(torch.float32).to(device)
            s_labels = s_labels.to(device)
            s_log_prop = s_log_prop.to(device)
            s_loss = s_loss.to(device)
            y = y.to(device)

            prob = hnet(X)

            logits = torch.cat((prob, 1-prob),dim=1)
            G_sample = gumbel_softmax(logits=logits, temperature=1, hard=is_gumbel_hard, cuda=is_cuda) # [bs, 2, n_label]
            G_sample = G_sample.view((G_sample.size(0), -1)) # [bs, 2*n_label]
            G_sample = torch.cat((G_sample,X),dim=-1)
            D_fake = Dnet_xy(G_sample)
            transformed_y = torch.cat((s_labels,1-s_labels),dim=1) # TODO: here
            transformed_y = transformed_y.view(transformed_y.size(0),-1)
            transformed_y = torch.cat((transformed_y, X),dim=-1).to(torch.float32)
            D_real = Dnet_xy(transformed_y)

            D_loss = -(torch.mean(D_fake) - torch.mean(0.25 * D_real ** 2 + D_real))
            D_loss.backward()
            opt_d.step()

            opt_d.zero_grad()
            opt_h.zero_grad()
            opt_h2.zero_grad()

            prob = hnet(X)

            logits = torch.cat((prob, 1-prob),dim=1)
            # generator h
            G_sample = gumbel_softmax(logits=logits, temperature=1, hard=is_gumbel_hard, cuda=is_cuda)  # [bs, 2, n_label]
            G_sample = G_sample.view((G_sample.size(0), -1))
            G_sample = torch.cat((G_sample, X), dim=-1)
            D_fake = Dnet_xy(G_sample)
            G_loss = torch.mean(D_fake)
            G_loss.backward()
            opt_h2.step()

            opt_d.zero_grad()
            opt_h.zero_grad()
            opt_h2.zero_grad()
