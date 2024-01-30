# -*- coding: utf-8 -*-
# @Time    : 2024/1/29 16:27
# @Author  : colagold
# @FileName: metrics.py

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
        self.bigger= True # 指标越大越好，如果为False，表示越小越好
    def __str__(self):
        return self.__class__.__name__


    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")

class RMSE(MetricAbstract):
    def __init__(self):
        self.bigger= False  #指标越小越好
    def __call__(self, groundtruth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        return np.linalg.norm(pred.data - groundtruth.data)/np.sqrt(groundtruth.nnz)


class MAE(MetricAbstract):
    def __init__(self):
        self.bigger= False  #指标越小越好

    def __call__(self, groundtruth, pred) -> float:
        assert sp.isspmatrix_csr(pred)
        assert sp.isspmatrix_csr(groundtruth)
        return np.mean(np.abs(pred.data - groundtruth.data))


class MSE(MetricAbstract):
    def __init__(self):
        self.bigger= False  #指标越小越好
    def __call__(self, groundtruth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        return np.mean(np.power(pred.data - groundtruth.data,2))


# Ranking-based

class NDCG5(MetricAbstract):
    def __init__(self):
        self.topk = 5

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # 没有物品交互记录的user不参与计算
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # 对所有物品预测评分进行排序,只取topk值及其对应的索引(item_id)
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)


class NDCG10(MetricAbstract):
    def __init__(self):
        self.topk = 10

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # 没有物品交互记录的user不参与计算
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            #print(f"positive item lens {len(pos_items)} all item lens {len(gt.data[uid])}")
            # logging.info(f"positive item lens {len(pos_items)} all item lens {len(gt.data[uid])}")
            # print(gt.data[uid])
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # 对所有物品预测评分进行排序,只取topk值及其对应的索引(item_id)
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)



class RECALL5(MetricAbstract):
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
        self.topk = 5

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # 没有物品交互记录的user不参与计算
        recall = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # 对所有物品预测评分进行排序,只取topk值及其对应的索引(item_id)
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            recall.append(sum(hits) / len(pos_items))
        return sum(recall) / len(recall)


class RECALL10(MetricAbstract):
    def __init__(self):
        self.topk = 10

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # 没有物品交互记录的user不参与计算
        recall = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            if len(pos_items)==0:continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # 对所有物品预测评分进行排序,只取topk值及其对应的索引(item_id)
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            recall.append(sum(hits) / len(pos_items))
        return sum(recall) / len(recall)



class Precision5(MetricAbstract):
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
        self.topk = 5

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # 没有物品交互记录的user不参与计算
        precision = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # 对所有物品预测评分进行排序,只取topk值及其对应的索引(item_id)
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            precision.append(sum(hits) / len(topk_items))  # 与recall/hr的区别
        return sum(precision) / len(precision)


class Precision10(MetricAbstract):
    def __init__(self):
        self.topk = 10

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # 没有物品交互记录的user不参与计算
        precision = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # 对所有物品预测评分进行排序,只取topk值及其对应的索引(item_id)
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            precision.append(sum(hits) / len(topk_items))  # 与recall/hr的区别
        return sum(precision) / len(precision)



