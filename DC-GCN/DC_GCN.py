
import copy
import io
import json
import math
import pickle
import random
import types
from time import time

import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import yaml
from scipy.sparse import csr_matrix
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, Tensor

from trainer import TrainerBase
import os
import  warnings
warnings.filterwarnings("ignore")

def MSELoss(groud_truth,pred):
    return nn.MSELoss()(groud_truth,pred)

class RSCompleteRatingDataSet:
    def __init__(self, ds:sp.csr_matrix):
        data = ds.tocoo()
        self.users = data.row
        self.items = data.col
        self.data = data.data

    def __len__(self):
        return len(self.users)

    def __getitem__(self, item):

        uid = self.users[item]
        iid = self.items[item]
        ratings = self.data[item]
        return uid,iid,ratings

class ModelAbstract:
    def __init__(self):
        self.checkpoint_path = None
        self.tensorboard_path=None
        self.cache_dir=None

        self.resume_state=True
    def train(self, ds,valid_ds = None,test_ds=None,valid_funcs=None,cb_progress=lambda x:None):

        return None

    def predict(self,ds,cb_progress=lambda x:None):

        return None

    def save(self,filepath:str=None):
        #  save model to a file
        return None

    def load(self,filepath:str=None):
        #  load model from a file
        return None

    def opt_one_batch(self, batch) -> dict:
        """

        """

    def eval_data(self, dataloader, metric, inbatch) -> float:
        """

        """

    def class_name(self):
        return str(self.__class__)[8:-2].split('.')[-1].lower()

    def __str__(self):
        parameters_dic=copy.deepcopy(self.__dict__)
        parameters=get_parameters_js(parameters_dic)
        return dict_to_yamlstr({self.class_name():parameters})

    def __getitem__(self, key):
        if isinstance(key,str) and hasattr(self,key):
            return getattr(self, key)
        else:
            return None
    def __setitem__(self, key,value):
        if isinstance(key,str):
            return setattr(self, key, value)
        else:
            return None

def dict_to_yamlstr(d:dict)->str:
    with io.StringIO() as mio:
        json.dump(d, mio)
        mio.seek(0)
        if hasattr(yaml, 'full_load'):
            y = yaml.full_load(mio)
        else:
            y = yaml.load(mio)
        return yaml.dump(y)  # 输出模型名字+参数

def get_parameters_js(js) -> dict:
    ans = None
    if isinstance(js, (dict)):
        ans = dict([(k,get_parameters_js(v)) for (k,v) in js.items() if not isinstance(v, types.BuiltinMethodType)])
    elif isinstance(js, (float, int, str)):
        ans = js
    elif isinstance(js, (list, set, tuple)):
        ans = [get_parameters_js(x) for x in js]
    elif js is None:
        ans = None
    else:
        ans = {get_full_class_name(js): get_parameters_js(js.__dict__)}
    return ans

def get_full_class_name(c)->str:
    s = str(type(c))
    return s[8:-2]  # remove "<classes '" 和 “'>”

class ExplicitRecAbstract(ModelAbstract):


    def predict(self, ds, cb_progress=lambda x: None):
        assert sp.isspmatrix_csr(ds)
        cb_progress(0)
        ds = ds.tocoo()
        uids = torch.from_numpy(ds.row)
        iids = torch.from_numpy(ds.col)

        # 将数据放入GPU
        uids = uids.to(self.device)
        iids = iids.to(self.device)

        pred = self.model(uids, iids)
        cb_progress(1.0)  # report progress
        data = pred.cpu().detach().numpy()
        return sp.csr_matrix((data, (ds.row, ds.col)), ds.shape)

    def opt_one_batch(self, batch) -> dict:
        """

        """

        uids, pos_iids, neg_iids = (x.to(self.device) for x in batch)

        pos_scores, neg_scores = self.model.get_rating(uids, pos_iids, neg_iids)
        loss = self.criterion(pos_scores, neg_scores)
        #print("BPR LOSS:", loss)
        # weight=torch.sigmoid_(self.weights[batch[1].long()]).to(self.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {}
        loss_dict['loss'] = float(loss.data.cpu().numpy())  # 不管数据在gpu还是cpu都统一存入cpu
        return loss_dict

    def eval_data(self, dataloader, metric) -> float:
        """

        """
        Y = []
        Pred = []
        users = []
        items = []

        for uids, iids, ratings in dataloader:
            uids = uids.to(self.device)
            iids = iids.to(self.device)
            pred = self.model(uids, iids)

            Y.append(ratings.cpu().numpy())
            Pred.append(pred.cpu().detach().numpy())
            users.append(uids.cpu().numpy())
            items.append(iids.cpu().numpy())

        Y = np.concatenate(Y)
        Pred = np.concatenate(Pred)
        users = np.concatenate(users)
        items = np.concatenate(items)

        Y = sp.csr_matrix((Y, (users, items)))
        Pred = sp.csr_matrix((Pred, (users, items)))

        return -metric(Y, Pred)  #

    def save(self, filepath=None):
        if filepath is None:  #
            filepath = self.checkpoint_path + ".last_model_state"
        with open(filepath, 'wb') as fout:
            state = {'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'torch_random_state': torch.random.get_rng_state().numpy(),
                     'numpy_random_state': np.random.get_state()}
            torch.save(state, fout)

    def load(self, filepath=None):
        if filepath is None:  #
            filepath = self.checkpoint_path + ".last_model_state"
        if not os.path.isfile(filepath):
            return

        with open(filepath, 'rb') as fin:
            state = torch.load(fin)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            random_state = torch.from_numpy(state['torch_random_state'])
            torch.random.set_rng_state(random_state)
            np.random.set_state(state['numpy_random_state'])

use_gpu = torch.cuda.is_available()
class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class dcgcn(BasicModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_layers = config["n_layers"]
        self.emb_dropout = config["emb_dropout"]
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.is_mlp = config["mlp"]
        self.agg = config["agg"]
        self.is_mask = config["is_mask"]
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=config["num_users"], embedding_dim=config["latent_dim"])
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=config["num_items"], embedding_dim=config["latent_dim"])
        self.embedding_rating = torch.nn.Embedding(
            num_embeddings=config["num_ratings"], embedding_dim=config["latent_dim"])
        self.embedding_rating.requires_grad = False

        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)
        nn.init.normal_(self.embedding_rating.weight, std=0.01)
        # world.cprint('use NORMAL distribution initilizer')

        self.dropout = nn.Dropout(p=config["emb_dropout"])


        self.user_t = nn.init.xavier_uniform_(torch.empty(self.num_users, 1))
        self.item_t = nn.init.xavier_uniform_(torch.empty(self.num_items, 1))
        self.user_t = nn.Parameter(self.user_t, requires_grad=True)
        self.item_t = nn.Parameter(self.item_t, requires_grad=True)



        self.rating_group_tensor = config["rating_group_dict"]

        self.aggre_rating = nn.Sequential(
            nn.Linear(in_features=config["latent_dim"] * 2, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=64)
        )

        self.predict_layer = nn.Sequential(
            nn.Linear(in_features=config["latent_dim"] * 2, out_features=64),
            nn.LeakyReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.LeakyReLU(),
            nn.Linear(in_features=32, out_features=1)
        )

        self.atten_weight_list = torch.nn.ModuleList()
        for _ in self.rating_group_tensor.keys():  # 不共享参数，两层的mlp
            self.atten_weight_list.append(
                nn.Sequential(
                    nn.Linear(in_features=config["latent_dim"] * 2, out_features=config["latent_dim"]),
                    nn.LeakyReLU(),
                    nn.Linear(in_features=config["latent_dim"], out_features=config["latent_dim"]),
                    # nn.LeakyReLU(),
                )
            )
        self.agg_layer = nn.Sequential(
            nn.Linear(in_features=config["latent_dim"] * 5, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=64)
        )
        self.dev = 'cpu'
        if torch.cuda.is_available():
            self.dev = 'cuda'

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        rating_emb = self.embedding_rating.weight

        embs = []

        for key, rating_tensor_graph in self.rating_group_tensor.items():
            key = int(key)

            all_embs = torch.cat([users_emb, items_emb], dim=0)

            t_rating_emb = rating_emb[int(key)].repeat(all_embs.shape[0], 1)

            t_cat = torch.cat([all_embs, t_rating_emb], dim=1)

            all_embs = self.atten_weight_list[int(key) - 1](t_cat)

            gcn_emb_per_rating = [all_embs]

            for layer in range(self.n_layers):
                if layer >= 2:
                    mask_ratio = 1 - math.log(1 / layer + 1)
                else:
                    mask_ratio = 0
                mask_tensor = self.generate_mask_matrix(self.config["latent_dim"], mask_ratio)
                mask_tensor = mask_tensor.to(self.config['device'])
                all_embs = torch.sparse.mm(rating_tensor_graph, all_embs)  # 卷积
                all_embs = torch.mm(all_embs, torch.diag(mask_tensor)) \
                           + torch.mm(gcn_emb_per_rating[-1], torch.diag(1 - mask_tensor))  # 掩码
                gcn_emb_per_rating.append(all_embs)

            rating_embs = torch.sum(torch.stack(gcn_emb_per_rating, dim=1), dim=1)
            embs.append(rating_embs)  #
        if self.agg == "mean":
            embs = torch.stack(embs, dim=1)
            out = torch.mean(embs, dim=1)
        elif self.agg == "sum":
            embs = torch.stack(embs, dim=1)
            out = torch.mean(embs, dim=1)
        elif self.agg == "mlp":
            embs = torch.cat(embs, dim=1)
            out = self.agg_layer(embs)
        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items

    def generate_mask_matrix(self, length, mask_rate):
        # 全1矩阵
        mask_matrix = torch.ones(1, length)

        # 计算掩码的数量
        num_masked = int(mask_rate * length)

        # 随机选择需要掩码的位置
        mask_indices = random.sample(range(length), num_masked)

        # 将选择的位置设置为0
        for idx in mask_indices:
            mask_matrix[0, idx] = 0

        return mask_matrix[0]

    def get_mask_matrix(self, mask_ratio, latent_dim):
        tensor = torch.ones(1, latent_dim)
        # 创建一个与输入tensor形状相同的随机0-1矩阵
        random_mask = torch.rand(tensor.shape)
        # 根据给定的概率生成置0的掩码
        zero_mask = random_mask < mask_ratio
        # 将tensor中对应位置置为0
        tensor[zero_mask] = 0
        return tensor[0]

    def predict(self, users, items):
        users_emb = self.embedding_user.weight[users.long()]
        items_emb = self.embedding_item.weight[items.long()]

        embd = torch.cat([users_emb, items_emb], dim=1)

        embd = self.predict_layer(embd)
        prediction = embd.flatten()
        # pred = (users_emb * items_emb).sum(1, keepdim=True)
        return prediction

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        pos_emb = all_items[items.long()]

        if self.is_mlp:
            embd = torch.cat([users_emb, pos_emb], dim=1)

            embd = self.predict_layer(embd)
            prediction = embd.flatten()
        else:
            prediction = (users_emb * pos_emb).sum(1)
        return prediction


class Weighted_MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: Tensor, target: Tensor, weights: Tensor) -> Tensor:
        return torch.mean(F.mse_loss(input, target, reduction='none').mul(weights))

class DC_GCN(ExplicitRecAbstract):
    def __init__(self):
        self.n_layers = 3
        self.latent_dim = 64
        self.batch_size = 4096
        self.lr = 0.001
        self.lambd = 0.01
        self.graph_noise = 0
        self.emb_dropout = 0
        self.rating_gate = 5
        self.nepochs = 10000
        self.mlp = True
        self.is_mask = True
        self.loss = "MSE"
        self.agg = "mlp"
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

    def train(self, ds, valid_ds=None, test_ds=None, valid_func=None, cb_progress=lambda x: None):
        assert sp.isspmatrix_csr(ds)
        self.num_users, self.num_items = ds.shape
        self.criterion = MSELoss
        # train_loader = DataLoader(self.DataSample(ds), batch_size=self.batch_size, shuffle=True)
        train_loader = DataLoader(RSCompleteRatingDataSet(ds), batch_size=self.batch_size, shuffle=True)
        if valid_ds is not None:
            valid_loader = DataLoader(RSCompleteRatingDataSet(valid_ds), batch_size=self.batch_size)
        else:
            valid_loader = None
        if test_ds is not None:
            test_loader = DataLoader(RSCompleteRatingDataSet(test_ds), batch_size=self.batch_size)
        else:
            test_loader = None

        #self.node_deg = get_node_deg(ds)


        self.rating_group_dict = self.split_rating_csr(ds, self.graph_noise)
        self.graph = self.getSparseGraph(ds)


        self.rating_graph = _convert_sp_mat_to_sp_tensor(ds).to(self.device)
        self.num_ratings = int(max(ds.data)) + 1
        # 按照评分划分，生成评分矩阵

        self.model = dcgcn(self.__dict__)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.lambd)
        #
        writer = SummaryWriter(self.tensorboard_path)  # 收集训练过程中的所有数据

        # 构建训练器，trainer，自动根据训练数据、验证数据和验证函数进行验证，并将中间过程记录到writer中
        # 需要注意的是：当前模型必须实现save，load，opt_one_batch， eval_data 函数
        trainer = TrainerBase(self.nepochs, valid_on_train_set=False)
        trainer.train(self, train_loader, valid_loader, test_loader, valid_func, writer)

    def getSparseGraph(self, ds):
        print("generating adjacency matrix")
        s = time()
        adj_mat = sp.dok_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        # R = self.UserItemNet.tolil()
        # R = ds.tolil()
        R = ds.tocoo()
        R.data[:] = 1
        adj_mat[:self.num_users, self.num_users:] = R
        adj_mat[self.num_users:, :self.num_users] = R.T
        adj_mat = adj_mat.todok()
        # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        # norm_adj = adj_mat.tocsr()
        norm_adj = norm_adj.tocsr()
        end = time()
        print(f"costing {end - s}s, saved norm_mat...")
        self.Graph = _convert_sp_mat_to_sp_tensor(norm_adj)
        self.Graph = self.Graph.coalesce().to(self.device)
        print("don't split the matrix")
        return self.Graph


    def predict(self,ds,cb_progress=lambda x:None):
        assert sp.isspmatrix_csr(ds)
        cb_progress(0)
        ds = ds.tocoo()
        uids = torch.from_numpy(ds.row)
        iids = torch.from_numpy(ds.col)
        if use_gpu:
            uids = uids.cuda()
            iids = iids.cuda()
        pred= self.model(uids, iids)

        #pred = self.model(ds.row, ds.col)
        cb_progress(1.0) # report progress
        if use_gpu:
            pred = pred.cpu()
        data = pred.detach().numpy()
        #data = pred.cpu().detach().numpy()
        return sp.csr_matrix((data,(ds.row,ds.col)),ds.shape)

    def opt_one_batch(self, batch) -> dict:
        """
        在batch数据上完成一次训练，并返回损失值，
        返回值为一个字典，其中必须包含”loss“关键字，代表了该batch数据下计算所得的损失值，如果需要返回其他值可以自行加入该字典
        返回到字典中的数据都将写入tensorboard中
        """

        # 在train函数中，利用RSCompleteRatingDataset对数据进行了转换，变成了 (uid，iid，rating)的元组，
        # 在经过torch的Dataloader将之转化为batch，则 batch的数据结构为：
        #              【(uid1,uid2,uid3,..), (iid1,iid2,iid3,...),(rating1,rating2,rating3,...)】

        # 先将batch中的所有数据放到模型指定的设备上，然后再取出
        uids, iids, ratings = (x.to(self.device) for x in batch)
        ratings = ratings.to(torch.float32)

        #loss = self.cal_neuralNDCG_loss(BPRLossMatrix ,self.model,uids,iids,ratings)#torch.mean(NDCGs)
        #loss = self.cal_bpr_loss(BPRLoss, self.model.predict, uids, iids, ratings)

        loss = self.cal_mse_loss(self.criterion, self.model, uids, iids, ratings)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_dict = {}
        loss_dict['loss'] = float(loss.data.cpu().numpy()) #不管数据在gpu还是cpu都统一存入cpu
        return loss_dict

    def cal_neuralNDCG_loss(self,loss_func,pred_func,uid,iid,ratings):

        ratings = ratings.to(torch.float32)
        #return loss_func(ratings,pred_func(uid,iid)[0])
        # #NDCG Loss Test
        # uid_1d = uids.view(-1)
        # iid_1d = iids.view(-1)

        nnz_idx = [np.nonzero(row)[0] for row in ratings.cpu().numpy()]
        # bounds = [ row[-1] for row in nnz_idx if len(row)>0]

        # 根据rattings中的非0值，预测pred矩阵中的值。
        t_nn_idx = torch.nonzero(ratings)
        row, col = t_nn_idx[:, 0], t_nn_idx[:, 1]
        nn_uids = uid[row, col]
        nn_iids = iid[row, col]
        nn_pred = pred_func(nn_uids, nn_iids)
        pred = torch.zeros_like(ratings)
        pred[row, col] = nn_pred

        loss = []
        for r, row in enumerate(nnz_idx):
            if len(row) > 0:
                bound = row[-1]
                per_loss = loss_func(pred[r:r + 1, :bound], ratings[r:r + 1, :bound])
                loss.append(per_loss)

        loss = sum(loss)  # torch.mean(NDCGs)
        return loss

    def cal_mse_loss(self,loss_func,pred_func,uid,iid,ratings):
        pred= pred_func(uid, iid)
        return loss_func(pred,ratings)


    def eval_data(self, dataloader, metric) -> float:
        """
        负责评估所训练的模型在数据集上的性能。
        根据传入的数据（可能是训练集，也可能是测试集，由模型的train函数决定），利用metric函数完成一次验证，并返回指标计算的结果。
        """
        Y = []
        Pred = []
        users  = [] #  因为 推荐算法的评价函数的输入是csr_matrix,这里需要记录用户id 和商品id，方便后面重构矩阵
        items = []

        for uids, iids, ratings in dataloader:
            uids = uids.to(self.device)
            iids = iids.to(self.device)
            pred= self.model(uids,iids)

            Y.append(ratings.cpu().numpy())
            Pred.append(pred.cpu().detach().numpy())
            users.append(uids.cpu().numpy())
            items.append(iids.cpu().numpy())

        Y = np.concatenate(Y)
        Pred = np.concatenate(Pred)
        users = np.concatenate(users)
        items = np.concatenate(items)

        Y = sp.csr_matrix((Y,(users,items)))
        Pred = sp.csr_matrix((Pred, (users, items)))
        return metric(Y, Pred)


    def _split_A_hat(self,A):
        A_fold = []
        fold_len = (self.n + self.m) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end = self.n + self.m
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(_convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to(self.device))
        return A_fold

    def split_rating_csr(self, ds,noise):
        ds = ds.tocoo()
        row = ds.row
        col = ds.col
        shape = ds.shape
        df = pd.DataFrame([row, col, ds.data], index=["user", "item", "rating"]).T
        df_group = df.groupby("rating")
        res = dict()

        def pd_merge(df):
            uids = df.iloc[:, 0].values
            iids = df.iloc[:, 1].values
            rates = df.iloc[:, 2].values
            return sp.csr_matrix((rates, (uids, iids)),
                                 dtype=np.float, shape=shape)

        for key, index in df_group.groups.items():
            csr=pd_merge(df[df.index.isin(index)])
            density=noise*csr.nnz/(shape[0]*shape[1])
            noise_csr=sp.rand(shape[0], shape[1], density=density, format='csr',random_state=1)
            csr=csr+noise_csr
            csr.data[:]=key
            res[key]=self.getSparseGraph(csr) .to(self.device) # 取出对应的DataFrame,转为CSR


        return res

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
