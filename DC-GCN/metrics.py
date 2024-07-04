import logging
import math

import numpy as np
import scipy.sparse as sp

import scipy.optimize as opt
from sklearn.metrics import normalized_mutual_info_score, \
    adjusted_rand_score, rand_score
import heapq


rating_gate=4.0

class MetricAbstract:
    def __init__(self):
        self.bigger= True
    def __str__(self):
        return self.__class__.__name__


    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")


class MAE(MetricAbstract):
    def __init__(self):
        self.bigger= False

    def __call__(self, groundtruth, pred) -> float:
        assert sp.isspmatrix_csr(pred)
        assert sp.isspmatrix_csr(groundtruth)
        return np.mean(np.abs(pred.data - groundtruth.data))


class MSE(MetricAbstract):
    def __init__(self):
        self.bigger= False
    def __call__(self, groundtruth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        return np.mean(np.power(pred.data - groundtruth.data,2))


# Ranking-based

class NDCG5(MetricAbstract):
    def __init__(self):
        self.topk = 5

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            if len(gt.data[uid])-len(pos_items)<5:continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)

class NDCG2(MetricAbstract):
    def __init__(self):
        self.topk = 2

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            if len(gt.data[uid]) - len(pos_items) < 5: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)

class NDCG1(MetricAbstract):
    def __init__(self):
        self.topk = 1

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            if len(gt.data[uid]) - len(pos_items) < 5: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)






class RECALL1(MetricAbstract):
    """ 召回率（Recall）, 表示所有的正例样本中，有多少被模型成功地找出. 和Hit@K计算方法一样
    正样本中被推荐的概率
    该指标越大越好。
     eg:
        pred/ground_truth:
            [1,2,3,4,7] / [2,7,9];  命中item 2, 7; RECALL_1 = 2/3
            [1,3,4,8,9] / [4,7];    命中item 4;   RECALL_2 = 1/2
        RECALL@5 = 1/2 * (2/3 + 1/2)
    """

    def __init__(self):
        self.topk = 1

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        recall = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            if len(gt.data[uid]) - len(pos_items) < 1: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            recall.append(sum(hits) / len(pos_items))
        return sum(recall) / len(recall)

class RECALL2(MetricAbstract):


    def __init__(self):
        self.topk = 2

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        recall = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            if len(gt.data[uid]) - len(pos_items) < 1: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            recall.append(sum(hits) / len(pos_items))
        return sum(recall) / len(recall)





class Precision1(MetricAbstract):
    """查准率（Precision），反映推荐列表在中item的查准率，强调预测的“准确性”;
    推荐的结果中属于正样本的概率
    计算公式: $1/N $
    该指标越大越好。
    eg:
        pred/ground_truth:
            [1,2,3,4,7] / [2,7,9];  命中item 2, 7; precision_1 = 2/5
            [1,3,4,8,9] / [4,7];    命中item 4;   precision_2 = 1/5
        precision@5 = 1/2 * (2/5 + 1/5)
    """

    def __init__(self):
        self.topk = 1

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        precision = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            if len(gt.data[uid]) - len(pos_items) < 5: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            precision.append(sum(hits) / len(topk_items))
        return sum(precision) / len(precision)

class Precision2(MetricAbstract):

    def __init__(self):
        self.topk = 2

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        precision = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            if len(gt.data[uid]) - len(pos_items) < 5: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            precision.append(sum(hits) / len(topk_items))
        return sum(precision) / len(precision)


