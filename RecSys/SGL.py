import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import scipy.sparse as sp
import random
from copy import deepcopy

class EdgeDrop(nn.Module):
    """ Drop edges in a graph.
    """
    def __init__(self, resize_val=False):
        super(EdgeDrop, self).__init__()
        self.resize_val = resize_val

    def forward(self, adj, keep_rate):
        """
        :param adj: torch_adj in data_handler
        :param keep_rate: ratio of preserved edges
        :return: adjacency matrix after dropping edges
        """
        if keep_rate == 1.0: return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newVals = vals[mask] / (keep_rate if self.resize_val else 1.0)
        newIdxs = idxs[:, mask]
        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class SGL (nn.Module):
    def __init__(self, data_generator, args):
        super(SGL, self).__init__()
        self.n_user = data_generator.n_users
        self.n_item = data_generator.n_items
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size
        self.args = args
        self.data = data_generator
        self.layers = args.layer_size

        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_item, self.emb_size)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.norm_adj  = data_generator.norm_adj
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(args.device)
        
        self.cl_rate = 0.01
        self.temp = 0.5
        print('cl rate = {}, temp  ={}'.format(self.cl_rate, self.temp))
        self.drop_out = EdgeDrop()

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
    

    def cal_infonce_loss(self, view1, view2, index):
        index = torch.unique(torch.Tensor(index).type(torch.long)).cuda()

        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
        
        view1_embs = view1[index]
        view2_embs = view2[index]

        view1_embs_abs = view1_embs.norm(dim=1)
        view2_embs_abs = view2_embs.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', view1_embs, view2_embs) / torch.einsum('i,j->ij', view1_embs_abs, view2_embs_abs)
        sim_matrix = torch.exp(sim_matrix / self.temp)
        pos_sim = sim_matrix[np.arange(view1_embs.shape[0]), np.arange(view1_embs.shape[0])]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        return loss.mean()
    
    def encoder (self, norm_adj):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        final_embeddings = torch.mean(torch.stack(all_embeddings, dim=1), dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_user, self.n_item])
        return user_all_embeddings, item_all_embeddings
    

    
    def forward (self, data, is_train):
        
        ue, ie = self.encoder(self.sparse_norm_adj)

        if is_train:
            pos = torch.sum(ue[data[0]] * ie[data[1]], dim=-1)
            neg = torch.sum(ue[data[0]] * ie[data[2]], dim=-1)
            loss = torch.mean(F.softplus(neg-pos))

            aug_norm_1 = self.drop_out(self.sparse_norm_adj, 0.9)
            aug_norm_2 = self.drop_out(self.sparse_norm_adj, 0.9)
            aug1_uEmbs, aug1_iEmbs = self.encoder(aug_norm_1)
            aug2_uEmbs, aug2_iEmbs = self.encoder(aug_norm_2)

            loss += self.cl_rate * (
                self.cal_infonce_loss(aug1_uEmbs, aug2_uEmbs, data[0]) +\
                self.cal_infonce_loss(aug1_iEmbs, aug2_iEmbs, data[1])
            )            
            return loss
        else:
            return torch.sum(ue[data[0]] * ie[data[1]], dim=-1)
        
