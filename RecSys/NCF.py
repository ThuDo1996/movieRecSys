import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import scipy.sparse as sp
import random

class NeuMF (nn.Module):
    def __init__(self, data_generator, args):
        super(NeuMF, self).__init__()
        self.n_user = data_generator.n_users
        self.n_item = data_generator.n_items
        self.emb_size_mf = args.emb_size_mf
        self.emb_size_mlp = args.emb_size_mlp
        self.args = args

        self.mf_user = nn.Embedding(self.n_user, self.emb_size_mf)
        self.mf_item = nn.Embedding(self.n_item, self.emb_size_mf)

        self.mlp_user = nn.Embedding(self.n_user, self.emb_size_mlp)
        self.mlp_item = nn.Embedding(self.n_item, self.emb_size_mlp)
        
        self.mlp_layers = nn.Sequential(
           nn.Linear(self.emb_size_mlp*2, self.emb_size_mlp),
           nn.ReLU(),


           nn.Linear(self.emb_size_mlp, self.emb_size_mlp//2),
           nn.ReLU(),

           nn.Linear(self.emb_size_mlp//2, self.emb_size_mlp//4),
           nn.ReLU()
        )

        self.predictor = nn.Sequential(nn.Linear(self.emb_size_mf + self.emb_size_mlp//4, 1), nn.Sigmoid())

        def init_weights(m):
            if isinstance(m, nn.Linear):
                print(m)
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
                
            if isinstance(m, nn.Embedding):
                print(m)
                torch.nn.init.xavier_uniform_(m.weight)

        self.apply(init_weights)
    
    def forward(self, data, is_train):
       user, item = data[0], data[1]
       u_mf = self.mf_user.weight
       u_mlp = self.mlp_user.weight
       i_mf = self.mf_item.weight
       i_mlp = self.mlp_item.weight

       mf_part = torch.mul(u_mf[user], i_mf[item])

       mlp_vec = torch.cat([u_mlp[user], i_mlp[item]], dim=-1)
       mlp_part = self.mlp_layers(mlp_vec)

       pred_vec = torch.cat([mf_part, mlp_part], dim=-1)
       pos_scores = self.predictor(pred_vec).flatten()
       
       if is_train:

            neg_item = data[2]
            mf_part = torch.mul(u_mf[user], i_mf[neg_item])

            mlp_vec = torch.cat([u_mlp[user], i_mlp[neg_item]], dim=-1)
            mlp_part = self.mlp_layers(mlp_vec)

            pred_vec = torch.cat([mf_part, mlp_part], dim=-1)
            neg_scores = self.predictor(pred_vec).flatten()

            loss_pred =  torch.mean(F.softplus(neg_scores-pos_scores))
            return loss_pred
       else:
           return pos_scores

       




        

    


