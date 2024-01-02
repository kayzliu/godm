import os
import tqdm
import math
import time
import copy
import pygod
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import BaseTransform, ToUndirected
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.utils import add_self_loops, negative_sampling

from .sage import SAGEConv
from .diffusion import MLPDiffusion, Model, sample_dm


class VGAE(nn.Module):
    def __init__(
            self,
            in_dim,
            hid_dim,
            temporal=True,
            t_min=0,
            t_max=1024,
            etypes=1,
            threshold=0.5
    ):
        super(VGAE, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.temporal = temporal
        self.t_min = t_min
        self.t_max = t_max
        self.time_len = self.t_max - self.t_min + 1 if temporal else None
        self.etypes = etypes
        self.threshold = threshold
        self.enc_shared = SAGEConv(in_dim, hid_dim, temporal=temporal,
                                   time_len=self.time_len, etypes=etypes)
        self.enc_mu = SAGEConv(hid_dim, hid_dim, temporal=temporal,
                               time_len=self.time_len, etypes=etypes)
        self.enc_sigma = SAGEConv(hid_dim, hid_dim, temporal=temporal,
                                  time_len=self.time_len, etypes=etypes)
        self.dec_attr = nn.Linear(hid_dim, in_dim)
        self.dec_time = nn.Linear(2 * hid_dim, 1)
        self.dec_type = nn.Linear(2 * hid_dim, self.etypes)
        self.dec_stru = nn.Linear(2 * hid_dim, 1)
        self.map_label_e = nn.Linear(1, in_dim, bias=False)
        self.map_label_d = nn.Linear(1, hid_dim, bias=False)

    def forward(self, x, pos_edge_index, neg_edge_index,
                label, edge_time=None, edge_type=None):
        self.mean, self.log_std = self.encode(x, pos_edge_index, label,
                                              edge_time, edge_type=edge_type)
        noise = torch.randn_like(self.mean)
        z = self.mean + noise * torch.exp(self.log_std)
        x_, edge_pred, t_, p_ = self.decode(z, pos_edge_index,
                                            neg_edge_index, label)
        return x_, edge_pred, t_, p_

    def encode(self, h, edge_index, label, edge_time=None, edge_type=None):
        h += self.map_label_e(label)

        h = self.enc_shared(h, edge_index, edge_time, edge_type)
        h = torch.relu(h)
        mean = self.enc_mu(h, edge_index, edge_time, edge_type)
        log_std = self.enc_sigma(h, edge_index, edge_time, edge_type)
        return mean, log_std

    def decode(self, z, pos_edge_index, neg_edge_index, label):
        z += self.map_label_d(label)
        x_ = self.dec_attr(z)

        pos_ze = torch.cat([z[pos_edge_index[0]], z[pos_edge_index[1]]], dim=1)
        neg_ze = torch.cat([z[neg_edge_index[0]], z[neg_edge_index[1]]], dim=1)

        pos_edge_pred = self.dec_stru(pos_ze).squeeze(-1)
        neg_edge_pred = self.dec_stru(neg_ze).squeeze(-1)
        edge_pred = torch.cat([pos_edge_pred, neg_edge_pred], dim=0)

        t_ = self.dec_time(pos_ze).squeeze(-1) if self.temporal else None
        p_ = self.dec_type(pos_ze) if self.etypes > 1 else None

        return x_, edge_pred, t_, p_

    def sample(self, z, label):
        z += self.map_label_d(label)
        x_ = self.dec_attr(z)

        z1 = z.unsqueeze(1).expand(-1, z.size(0), -1)
        z2 = z.unsqueeze(0).expand(z.size(0), -1, -1)
        ze = torch.cat((z1, z2), dim=2)

        adj = torch.sigmoid(self.dec_stru(ze)).squeeze(-1)
        edge_index = (adj > self.threshold).nonzero().T
        edge_index = add_self_loops(edge_index, num_nodes=z.size(0))[0]

        pos_ze = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        if self.temporal:
            t_ = self.dec_time(pos_ze).squeeze(-1)
            t_ = torch.clamp(t_, min=0, max=1)
            t_ = t_ * (self.t_max - self.t_min)
        else:
            t_ = None
        p_ = self.dec_type(pos_ze).argmax(-1) if self.etypes > 1 else None

        return x_, edge_index, t_, p_


class GODM(BaseTransform):
    def __init__(self,
                 name="",
                 hid_dim=None,
                 diff_dim=None,
                 vae_epochs=100,
                 diff_epochs=100,
                 patience=50,
                 lr=0.001,
                 wd=0.,
                 batch_size=2048,
                 threshold=0.75,
                 wx=1.,
                 we=.5,
                 beta=1e-3,
                 wt=1.,
                 time_attr='edge_time',
                 type_attr='edge_type',
                 wp=.3,
                 gen_nodes=None,
                 sample_steps=50,
                 device=0,
                 verbose=False):

        self.name = name
        self.hid_dim = hid_dim
        self.diff_dim = diff_dim
        self.vae_epochs = vae_epochs
        self.diff_epochs = diff_epochs
        self.patience = patience
        self.lr = lr
        self.wd = wd
        self.batch_size = batch_size
        self.threshold = threshold
        self.wx = wx
        self.we = we
        self.beta = beta
        self.time_attr = time_attr
        self.temporal = True
        self.wt = wt
        self.type_attr = type_attr
        self.etypes = 1
        self.wp = wp
        self.gen_nodes = gen_nodes
        self.sample_steps = sample_steps
        self.device = pygod.utils.validate_device(device)
        self.verbose = verbose

        self.ae = None
        self.dm = None

        self.y0 = None
        self.emap = None
        self.mean = None
        self.std = None
        self.t_min = None
        self.t_max = None

    def __call__(self, data):
        self.arg_parse(data)
        data = self.preprocess(data)

        cluster_data = ClusterData(data,
                                   num_parts=data.num_nodes // self.batch_size,
                                   log=self.verbose)
        dataloader = ClusterLoader(cluster_data, batch_size=1, shuffle=True,
                                   num_workers=4)

        if not os.path.exists('ckpt'):
            os.mkdir('ckpt')

        self.ae = VGAE(
            data.num_node_features,
            self.hid_dim,
            temporal=self.temporal,
            t_min=self.t_min,
            t_max=self.t_max,
            etypes=self.etypes,
            threshold=self.threshold
        ).to(self.device)

        self.train_ae(dataloader)

        denoise_fn = MLPDiffusion(self.hid_dim, self.diff_dim).to(self.device)
        self.dm = Model(denoise_fn=denoise_fn,
                        hid_dim=self.hid_dim).to(self.device)

        self.train_dm(dataloader)

        gen_gs = []
        gen_nodes = self.gen_nodes
        while gen_nodes > self.batch_size:
            gen_gs.append(self.sample(self.dm, self.batch_size))
            gen_nodes -= self.batch_size
        gen_gs.append(self.sample(self.dm, gen_nodes))

        data.y = self.y0

        aug_data = Batch.from_data_list([data] + gen_gs)

        aug_data = self.postprocess(aug_data)
        torch.save(aug_data, 'ckpt/' + self.name + '_aug_data.pt')

        return aug_data

    def arg_parse(self, data):
        if not isinstance(data, Data):
            raise TypeError('data must be torch_geometric.data.Data')

        if not hasattr(data, 'x'):
            raise ValueError('data must have feature x')

        if not hasattr(data, 'y'):
            raise ValueError('data must have label y')

        if self.hid_dim is None:
            self.hid_dim = 2 ** int(math.log2(data.x.size(1)) - 1)
        if self.diff_dim is None:
            self.diff_dim = 2 * self.hid_dim

        # mask out val and test nodes in training
        self.y0 = copy.deepcopy(data.y)
        data.y[data.train_mask == 0] = 0

        if not hasattr(data, self.time_attr):
            self.temporal = False
            self.wt = 0.

        if hasattr(data, self.type_attr):
            self.etypes = getattr(data, self.type_attr).unique().size(0)

        if self.gen_nodes is None:
            self.gen_nodes = data.y[data.train_mask].sum()

    def preprocess(self, data):
        # to undirected
        if data.is_directed():
            data = ToUndirected(reduce='min')(data)

        # normalize node feature
        self.mean, self.std = data.x.mean(0), data.x.std(0)
        data.x = (data.x - self.mean) / self.std

        # reindex edge type
        if self.etypes > 1:
            edge_type = getattr(data, self.type_attr)
            self.emap, edge_type = edge_type.unique(return_inverse=True)
            setattr(data, self.type_attr, edge_type)

        # time range
        if self.temporal:
            edge_time = getattr(data, self.time_attr)
            self.t_min, self.t_max = edge_time.min(), edge_time.max()
            setattr(data, self.time_attr, (edge_time - self.t_min))

        return data

    def postprocess(self, data):
        # denormalize
        data.x = data.x * self.std + self.mean

        # recover edge type
        if self.etypes > 1:
            edge_type = getattr(data, self.type_attr)
            setattr(data, self.type_attr, self.emap[edge_type])

        # recover time
        if self.temporal:
            edge_time = getattr(data, self.time_attr)
            setattr(data, self.time_attr, edge_time + self.t_min)

        return data

    def train_ae(self, dataloader):
        if self.verbose:
            print('Training autoencoder ...')
        optimizer = torch.optim.Adam(self.ae.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.wd)

        best_loss = float('inf')
        patience = 0
        for epoch in range(self.vae_epochs):
            start = time.time()
            self.ae.train()
            total_loss = 0
            num_nodes = 0
            for batch in dataloader:
                batch_size = batch.x.size(0)
                x = batch.x.to(self.device)
                pos_edge_index = batch.edge_index.to(self.device)
                neg_edge_index = negative_sampling(
                    edge_index=pos_edge_index,
                    num_nodes=batch_size)

                edge_label = torch.cat([torch.ones_like(pos_edge_index[0]),
                                        torch.zeros_like(neg_edge_index[0])])

                y = batch.y.float().unsqueeze(1).to(self.device)

                t = getattr(batch, self.time_attr).to(self.device) \
                    if self.temporal else None
                p = getattr(batch, self.type_attr).to(self.device) \
                    if self.etypes > 1 else None

                x_, edge_pred, t_, p_ = self.ae(x, pos_edge_index,
                                                neg_edge_index, y, t, p)
                loss = self.recon_loss(x, x_, edge_label.float(), edge_pred,
                                       t, t_, p, p_)

                kl_div = (0.5 / x_.size(0) *
                          (1 + 2 * self.ae.log_std - self.ae.mean ** 2 -
                           torch.exp(self.ae.log_std) ** 2).sum(1).mean())
                loss -= self.beta * kl_div

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * batch_size
                num_nodes += batch_size

            curr_loss = total_loss / num_nodes

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(self.ae, 'ckpt/' + self.name + '_ae.pt')
            else:
                patience += 1
                if patience == self.patience:
                    if self.verbose:
                        print('Early stopping')
                    break

            epoch_t = time.time() - start
            if self.verbose:
                print(f'Epoch: {epoch:03d}, Loss: {curr_loss:.6f}, '
                      f'Time: {epoch_t:.4f}')

        self.ae = torch.load('ckpt/' + self.name + '_ae.pt')

    def train_dm(self, dataloader):
        if self.verbose:
            print('Training diffusion model ...')
        optimizer = torch.optim.Adam(self.dm.parameters(), lr=self.lr,
                                     weight_decay=self.wd)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                      patience=20, verbose=self.verbose)

        self.dm.train()
        best_loss = float('inf')
        patience = 0
        for epoch in range(self.diff_epochs):
            pbar = tqdm.tqdm(dataloader, total=len(dataloader),
                             disable=not self.verbose)
            pbar.set_description(f"Epoch {epoch}")

            batch_loss = 0.0
            len_input = 0
            for batch in pbar:
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                y = batch.y.float().unsqueeze(1).to(self.device)
                t = getattr(batch, self.time_attr).to(self.device) \
                    if self.temporal else None
                p = getattr(batch, self.type_attr).to(self.device) \
                    if self.etypes > 1 else None

                inputs, _ = self.ae.encode(x, edge_index, y, t, p)
                loss = self.dm(inputs, y)

                loss = loss.mean()

                batch_loss += loss.item() * len(inputs)
                len_input += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.dm.parameters(), 1.0)
                optimizer.step()

                pbar.set_postfix({"Loss": loss.item()})

            curr_loss = batch_loss / len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience = 0
                torch.save(self.dm, 'ckpt/' + self.name + '_dm.pt')
            else:
                patience += 1
                if patience == self.patience:
                    if self.verbose:
                        print('Early stopping')
                    break

        self.dm = torch.load('ckpt/' + self.name + '_dm.pt')

    def sample(self, model, graph_size):
        net = model.denoise_fn_D
        noise = torch.randn(graph_size, self.hid_dim).to(self.device)
        label = torch.ones(graph_size).unsqueeze(1).to(self.device)

        if self.sample_steps > 0:
            z = sample_dm(net, noise, label, self.sample_steps)
        else:
            z = noise
        x_, edge_index, t_, p_ = self.ae.sample(z, label)

        data = Data(x=x_, edge_index=edge_index,
                    y=label.squeeze().long(), edge_time=t_,
                    edge_type=p_).cpu().detach()
        if self.temporal:
            setattr(data, self.time_attr, t_.cpu().detach())
        if self.etypes > 1:
            setattr(data, self.type_attr, p_.cpu().detach())
        data.train_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        return data

    def recon_loss(self, x, x_, edge_label, edge_pred,
                   t=None, t_=None, p=None, p_=None):
        loss_x = F.mse_loss(x_, x)
        if self.verbose:
            print("Batch Loss: feature: {:.4f}".format(loss_x.item()), end=' ')
        loss_e = F.binary_cross_entropy_with_logits(edge_pred, edge_label)
        if self.verbose:
            print("time: {:.4f}".format(loss_e.item()), end=' ')
        if self.temporal:
            loss_t = F.mse_loss(t_, t / (self.t_max - self.t_min))
            if self.verbose:
                print("time: {:.4f}".format(loss_t.item()), end=' ')
        else:
            loss_t = 0
        if self.etypes > 1:
            loss_p = F.cross_entropy(p_, p)
            if self.verbose:
                print("type: {:.4f}".format(loss_p.item()), end=' ')
        else:
            loss_p = 0
        loss = (self.wx * loss_x + self.we * loss_e +
                self.wt * loss_t + self.wp * loss_p)
        if self.verbose:
            print()
        return loss
