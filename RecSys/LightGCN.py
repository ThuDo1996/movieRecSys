import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import scipy.sparse as sp
import random

class LightGCN (nn.Module):
    def __init__(self, data_generator, args):
        super(LightGCN, self).__init__()
        self.n_user = data_generator.n_users
        self.n_item = data_generator.n_items
        self.emb_size = args.emb_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        
        self.data = data_generator
        self.layers = args.layer_size

    # init the weight of user-item
        self.user_embedding = nn.Embedding(self.n_user, self.emb_size)
        self.item_embedding = nn.Embedding(self.n_item, self.emb_size)
        
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        self.norm_adj, _ = data_generator.load_graph()
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(args.device)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape)).coalesce()

    
    def encoder (self):
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], 0)
        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)

        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings, [self.n_user, self.n_item])
        return user_all_embeddings, item_all_embeddings
    
    def forward (self, data, is_train):
        
        ue, ie = self.encoder()

        if is_train:
            pos = torch.sum(ue[data[0]] * ie[data[1]], dim=-1)
            neg = torch.sum(ue[data[0]] * ie[data[2]], dim=-1)
            loss = torch.mean(F.softplus(neg-pos))
            return loss
        else:
            return torch.sum(ue[data[0]] * ie[data[1]], dim=-1)
        
