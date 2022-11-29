import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_DoseFormer.pyGAT import GAT
import torch
import torch.nn as nn
import numpy as np
import copy
from graph_transformer_pytorch import GraphTransformer


class CNN1DEncoder(torch.nn.Module):
    def __init__(self):
        super(CNN1DEncoder, self).__init__()
        self.layer1 = nn.Conv1d(7, 80, padding=2, kernel_size=(3,), dtype=torch.float)
        self.layer2 = nn.Conv1d(80, 80, padding=2, kernel_size=(3,), dtype=torch.float)
        self.layer3 = nn.Conv1d(80, 80, padding=2, kernel_size=(3,), dtype=torch.float)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.node_number = 0

    def forward(self, x):
        self.node_number = int(x.shape[0]/6)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(self.node_number,6,80)
        return x

class cnn_lstm_attention(nn.Module):
    def __init__(self):
        super(cnn_lstm_attention, self).__init__()
        self.cnn = CNN1DEncoder()
        self.rnn = nn.LSTM(
            input_size=80, 
            hidden_size=80, 
            batch_first=True,
            bidirectional=True
            )
        self.hid = nn.Linear(129,160)
        self.cel = nn.Linear(129,160)
        self.node_number = 0
    def forward(self, x,static):
        x = self.cnn(x)
        self.node_number = self.cnn.node_number
        h = self.hid(static).view(self.node_number,2,80).permute(1, 0, 2).contiguous()
        c = self.cel(static).view(self.node_number,2,80).permute(1, 0, 2).contiguous()
        
        out ,(hidden,cell) = self.rnn(x,(h,c))
        
       
        return out


class cnn_lstm_attention_gt(nn.Module):
    def __init__(self):
        super(cnn_lstm_attention_gt, self).__init__()
        self.cnn_lstm = cnn_lstm_attention()
        # self.gnn = GAT(
        #     nfeat = 512,
        #     nhid = 512,
        #     nclass = 2,
        #     dropout = 0.5,
        #     nheads = 8,
        #     alpha = 0.2,
        # )
        self.gnn = GraphTransformer(
            dim = 160,
            depth = 2,
            edge_dim = 1,             
            with_feedforwards = True,   
            gated_residual = True,     
            rel_pos_emb = True          
        )
        self.graph_th = 1/2**0.5
        self.w_omega = nn.Parameter(torch.Tensor(
            2*80 , 2*80 ))
        self.u_omega = nn.Parameter(torch.Tensor(2*80 , 1))
        

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

        self.fc1 = nn.Linear(160,2)

        

    def makeadj(self,p):
        pdot = torch.matmul(p,p.permute(1,0))
        psqu = torch.sum(p**2,dim=1)
        psqrt = psqu**0.5
        pfr =torch.matmul(psqrt.view(psqrt.shape[0],-1),psqrt.view(-1,psqrt.shape[0])) 
        pcos = pdot / pfr
        zero_vec = torch.zeros_like(pcos)
        pdo_n = torch.where(pcos > self.graph_th , pcos, zero_vec)
        pdo_n_s = 1/torch.sum(pdo_n,axis=1)**0.5
        pdi = torch.diag(pdo_n_s)
        dad = torch.matmul(pdi , torch.matmul(pdo_n,pdi))
        return dad
    def forward(self, xss,static):

        outs = self.cnn_lstm(xss,static)
        x = outs
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束
        
        feat = torch.sum(scored_x, dim=1) #加权求和

        outs = feat

        adj = self.makeadj(outs)
        
        outs = outs.view(1,self.cnn_lstm.node_number,160)
        adj = adj.view(1,self.cnn_lstm.node_number,self.cnn_lstm.node_number,1)
        

        gout, edges = self.gnn(outs,adj)

        gout = gout.view(-1,160)

        out = self.fc1(gout)
        

        out = F.log_softmax(F.elu(out),dim=1)


        return out,adj



