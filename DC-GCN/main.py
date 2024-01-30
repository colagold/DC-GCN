import copy
import importlib
import io
import os
import pickle

import pandas as pd
import scipy.sparse as sp
import numpy as np
import zipfile
import yaml
from collections import Counter
from metrics import MAE,NDCG5,NDCG10,RECALL5,RECALL10,Precision5,Precision10


data_dir='data/Amazon_CDs_and_Vinyl10.zip'
model_config='DC_GCN.yaml'
data_name=data_dir.split("/")[-1].split(".")[0]
log_path=f"log/{data_name}"
if not os.path.isdir(log_path):
    os.makedirs(log_path)
checkpoint_dir = os.path.join(log_path,'check_point')

def set_seed(seed=2333):
    # print("SET SEED is Called ")
    # try:
    #     import torch
    #     torch.manual_seed(seed)
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed_all(seed)
    #         torch.backends.cudnn.deterministic = True
    #         torch.backends.cudnn.benchmark = False
    # except Exception as e:
    #     print("Set seed failed,details are ", e)
    #     pass
    # import numpy as np
    # np.random.seed(seed)
    # import random as python_random
    # python_random.seed(seed)

    import random,os, torch, numpy as np
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def smart_convert(value):
    assert isinstance(value, str)
    if value.count('.') > 0:
        try:
            return float(value)
        except:
            pass
    try:  # 检查整数
        return int(value)
    except:
        return value


def need_import(value):

    if isinstance(value, str) and len(value) > 3 and value[0] == value[-1] == '_' and not value == "__init__":
        return True
    else:
        return False


def create_obj_from_json(js):


    if isinstance(js, dict): #传入是字典
        rtn_dict = {}
        for key, values in js.items():
            if need_import(key):
                assert values is None or isinstance(values,
                                                    dict), f"拟导入的对象{key}的值必须为dict或None，用于初始化该对象"
                assert len(js) == 1, f"{js} 中包含了需要导入的{key}对象，不能再包含其他键值对"
                key = key[1:-1]  # 去掉 key的前后 `_`
                cls = my_import(key)
                if "__init__" in values:
                    assert isinstance(values, dict), f"__init__ 关键字，放入字典对象，作为父类{key}的初始化函数"
                    init_params = create_obj_from_json(values['__init__'])
                    if isinstance(init_params, dict):
                        obj = cls(**init_params)
                    else:
                        obj = cls(init_params)
                    values.pop("__init__")
                else:
                    obj = cls()
                # 此处已经不包含 "__init__"的key，value对
                for k, v in values.items():
                    setattr(obj, k, create_obj_from_json(v))
                return obj
            rtn_dict[key] = create_obj_from_json(values)
        return rtn_dict
    elif isinstance(js, (set, list)):
        return [create_obj_from_json(x) for x in js]
    elif isinstance(js,str):
        if need_import(js):
            cls_name = js[1:-1]
            return my_import(cls_name)()
        else:
            return js
    else:
        return js


def my_import(name):
    components = name.split('.')
    model_name = '.'.join(components[:-1])
    class_name = components[-1]
    mod = importlib.import_module(model_name)
    cls = getattr(mod, class_name)
    return cls


def myloads(jstr):
    if hasattr(yaml, 'full_load'):
        js = yaml.full_load(io.StringIO(jstr))
    else:
        js = yaml.load(io.StringIO(jstr))
    if isinstance(js, str):
        return {js: {}}
    else:
        return js



def update_parameters(param: dict, to_update: dict) -> dict:
    for k, v in param.items():
        if k in to_update:
            if to_update[k] is not None:
                if isinstance(param[k], (dict,)):
                    param[k].update(to_update[k])
                else:
                    param[k] = to_update[k]
            to_update.pop(k)
    param.update(to_update)
    return param


def str_obj_js(mystr):

    if ':' in mystr:
        return myloads(mystr)
    else:
        return {mystr: {}}


def enclose_class_name(value):
    if isinstance(value,dict):
        assert len(value)==1, "only one class"
        for k,v in value.items():
            if k[0]==k[-1]=="_":
                return {k:v}
            else:
                return {f"_{k}_":v}
    elif isinstance(value,str):

        if value[0]==value[-1]=="_":
            return value
        else:
            return f"_{value}_"
    else:
        return value
def parse_objects(filedict):
    algorithm = create_obj_from_json(enclose_class_name({filedict['algorithm']:filedict['algorithm_parameters']}))
    protocol = create_obj_from_json(enclose_class_name(filedict['protocol']))
    scenario = create_obj_from_json(enclose_class_name(filedict['scenario']))
    metrics = []
    for m in filedict['metrics']:
        metrics.append(create_obj_from_json(enclose_class_name(m)))

    return scenario, protocol, algorithm, metrics

def parse_data(data_path):

    df = None
    if os.path.isfile(data_path):
        z = zipfile.ZipFile(data_path, "r")
        for filename in z.namelist():
            if filename == 'ratings.csv':
                df = pd.read_csv(z.open(filename), header=None)
        assert df is not None, "not found ratings.csv文件"

    else:
        df = pd.read_table(data_path + "/ratings.csv", header=None)


    uids = df.iloc[:, 0].values
    iids = df.iloc[:, 1].values
    rates = df.iloc[:, 2].values
    ds = sp.csr_matrix((rates, (uids, iids))
                       ,dtype=np.float32)
    return ds

def merge( datalist,data_shape):
    data = datalist[0]
    for d in datalist:
        if d == data: continue
        data.extend(d)

    data = list(zip(*data))
    return sp.csr_matrix((data[2], (data[0], data[1])), shape=data_shape)


def get_noise_train_csr(n_user,coo,train_set,validate_set,test_set,split_arrays):
    all_items = set(coo.col)
    tr_noise_samples = ([], [], [])
    construct_train_data = ([], [], [])
    cold_start_data = [([], [], []), ([], [], []), ([], [], [])]
    csr_data=train_set
    train_lil=train_set.tolil()
    np.random.seed(1)
    ratings = set(train_set.data)
    for u in range(n_user):
        tr_items = set(train_lil.rows[u])
        # 采样
        if u in split_arrays[0]:
            construct_tr_item = np.random.choice(list(tr_items), size=np.random.randint(1,6), replace=False)
            construct_train_data[0].extend([u] * len(construct_tr_item))
            construct_train_data[1].extend(construct_tr_item)
            for i in construct_tr_item:
                construct_train_data[2].append(csr_data[u, i])

            cold_test_item = list(tr_items - set(construct_tr_item))
            cold_start_data[0][0].extend([u] * len(cold_test_item))
            cold_start_data[0][1].extend(cold_test_item)
            for i in cold_test_item:
                cold_start_data[0][2].append(csr_data[u, i])

        if u in split_arrays[1]:
            construct_tr_item = np.random.choice(list(tr_items), size=np.random.randint(6,11), replace=False)
            construct_train_data[0].extend([u] * len(construct_tr_item))
            construct_train_data[1].extend(construct_tr_item)
            for i in construct_tr_item:
                construct_train_data[2].append(csr_data[u, i])

            cold_test_item = list(tr_items - set(construct_tr_item))
            cold_start_data[1][0].extend([u] * len(cold_test_item))
            cold_start_data[1][1].extend(cold_test_item)
            for i in cold_test_item:
                cold_start_data[1][2].append(csr_data[u, i])

        if u in split_arrays[2]:
            construct_tr_item = np.random.choice(list(tr_items), size=np.random.randint(16,21), replace=False)
            construct_train_data[0].extend([u] * len(construct_tr_item))
            construct_train_data[1].extend(construct_tr_item)
            for i in construct_tr_item:
                construct_train_data[2].append(csr_data[u, i])

            cold_test_item = list(tr_items - set(construct_tr_item))
            cold_start_data[2][0].extend([u] * len(cold_test_item))
            cold_start_data[2][1].extend(cold_test_item)
            for i in cold_test_item:
                cold_start_data[2][2].append(csr_data[u, i])

        val_items = set(validate_set.rows[u])
        te_items = set(test_set.rows[u])
        if len(tr_items) == 0:
            continue
        if len(val_items) == 0 and len(te_items) == 0:
            continue  #

        neg_items = all_items - val_items - te_items - tr_items
        sample_num = min(len(neg_items), 0)
        if sample_num == 0: continue
        noise_users = [u] * sample_num
        if len(te_items) > 0:
            noise_item = np.random.choice(list(neg_items), sample_num, replace=False)
            tr_noise_samples[0].extend(noise_users)
            tr_noise_samples[1].extend(noise_item)
            tr_noise_samples[2].extend(np.random.choice(list(ratings), size=sample_num, replace=True))


    tr_noise_set = sp.csr_matrix((tr_noise_samples[2], (tr_noise_samples[0], tr_noise_samples[1])), csr_data.shape,
                                 dtype=np.float32)
    sample_csr = sp.csr_matrix((construct_train_data[2], (construct_train_data[0], construct_train_data[1])),
                               csr_data.shape, dtype=np.float32)
    return tr_noise_set,sample_csr,cold_start_data

def get_bias_group(n_item:int,train_item_count:dict,test_list:list,data_shape):
    sample_num = int(n_item * 0.1)
    head, tail = list(train_item_count.keys())[:sample_num], list(train_item_count.keys())[
                                                                    sample_num:]
    bias_group = [[], []]
    for i in test_list:
        if i[1] in head:
            bias_group[0].append(i)
        if i[1] in tail:
            bias_group[1].append(i)

    for i, data in enumerate(bias_group):
        #logging.info(f"bias interaction num:{len(data)}")
        bias_group[i] = merge([data],data_shape)
    return bias_group

def split(data) -> tuple:

    # data.data[:]=1
    data = data.tocoo()
    csr_data=data.tocsr()
    data01 = copy.deepcopy(data)
    data_shape = data.shape


    entries = list(zip(data01.row, data01.col, data01.data))
    np.random.seed(1)
    np.random.shuffle(entries)
    N = len(entries)
    train_percent=0.8
    valid_percent=0.1


    train_list = entries[:int(N * train_percent)]
    test_list = entries[
                int(N * train_percent):int(N * (train_percent + valid_percent))]  # 10%测试：[0.8,0.9]
    valid_list = entries[int(N * (train_percent + valid_percent)):]




    coo=data

    train_set = merge([train_list],data_shape).tolil()
    validate_set = merge([valid_list],data_shape).tolil()
    test_set = merge([test_list],data_shape).tolil()
    origion_train_set=train_set.tocsr()
    origion_test_set=test_set.tocsr()
    origion_val_set=validate_set.tocsr()

    n_user, n_item = coo.shape
    train_user_intera_count=dict(Counter(train_set.tocoo().row).most_common(n_user))
    train_item_intera_count = dict(Counter(train_set.tocoo().col).most_common(n_item))


    construct_user=int(n_user*0.25)
    sample_head_users=np.random.choice(list(train_user_intera_count.keys())[0:construct_user],
                                replace=False,
                                size=int(construct_user*0.5))
    split_points = [len(sample_head_users) // 3, 2 * len(sample_head_users) // 3]
    split_arrays = np.split(sample_head_users, split_points)

    new_train_list=[e for e in train_list if e[0] not in sample_head_users]

    tr_noise_set,sample_csr,cold_start_data=get_noise_train_csr( n_user, coo, origion_train_set, validate_set, test_set, split_arrays)


    del_tr_csr=merge([new_train_list],data_shape)

    train_set =del_tr_csr+sample_csr + tr_noise_set


    for i,e in enumerate(cold_start_data):
        cold_start_data[i]=sp.csr_matrix((e[2],(e[0],e[1])),data.shape,dtype=np.float32)

    bias_group=get_bias_group(n_item,train_item_intera_count,test_list,data_shape)


    return train_set, origion_val_set, origion_test_set,cold_start_data,bias_group

def main():
    set_seed()
    model_config = 'DC_GCN.yaml'
    with open(model_config, 'rb') as infile:
        cfg = yaml.safe_load(infile)

    data_name = cfg["data_path"].split("/")[-1].split(".")[0]
    log_path = f"log/{data_name}"
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    checkpoint_dir = os.path.join(log_path, 'check_point')

    data = parse_data(cfg["data_path"])

    process_path = f"process_data/"
    if not os.path.isdir(process_path):
        os.makedirs(process_path)
    process_data_path = os.path.join(process_path,data_name + ".pkl")
    if not os.path.isfile(process_data_path):
        train_data, valid_data, test_data, cold_start_group, popularity_group = split(data)
        process_data = {
            "train_data": train_data,
            "test_data": test_data,
            "valid_data": valid_data,
            "cold_group": cold_start_group,
            "popularity_group": popularity_group
        }
        with open(process_data_path, 'wb') as f:
            pickle.dump(process_data, f)
    else:
        with open(process_data_path, 'rb') as f:
            data = pickle.load(f)
            train_data = data["train_data"]
            test_data = data["test_data"]
            valid_data = data["valid_data"]
            cold_start_group = data["cold_group"]
            popularity_group = data["popularity_group"]

    train_set, origion_val_set, origion_test_set= train_data,valid_data,test_data
    metrics = [MAE(),NDCG5(),NDCG10(),RECALL5(),RECALL10(),Precision5(),Precision10()]
    valid_funs = metrics



    algorithm = create_obj_from_json(enclose_class_name({cfg['algorithm']: cfg['algorithm_parameters']}))
    algorithm.checkpoint_path = checkpoint_dir
    algorithm.tensorboard_path=log_path
    algorithm.train(train_set, origion_val_set, origion_test_set, valid_funs[0])
    pred = algorithm.predict(origion_test_set)
    results = [m(origion_test_set, pred) for m in metrics]
    headers = [str(m) for m in metrics]
    print(dict(zip(headers, results)))


if __name__=="__main__":
    main()