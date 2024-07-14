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
from metrics import *

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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  #
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
                                                    dict), f"dict or None"
                assert len(js) == 1, f""
                key = key[1:-1]  # 去掉 key的前后 `_`
                cls = my_import(key)
                if "__init__" in values:
                    assert isinstance(values, dict), f"__init__ "
                    init_params = create_obj_from_json(values['__init__'])
                    if isinstance(init_params, dict):
                        obj = cls(**init_params)
                    else:
                        obj = cls(init_params)
                    values.pop("__init__")
                else:
                    obj = cls()
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


def merge( datalist,data_shape):
    data = datalist[0]
    for d in datalist:
        if d == data: continue
        data.extend(d)

    data = list(zip(*data))
    return sp.csr_matrix((data[2], (data[0], data[1])), shape=data_shape)



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

    process_path = f"process_data/"
    process_data_path = os.path.join(process_path,data_name + ".pkl")


    with open(process_data_path, 'rb') as f:
        data = pickle.load(f)
        train_data = data["train_data"]
        test_data = data["test_data"]
        valid_data = data["valid_data"]

    train_set, origion_val_set, origion_test_set= train_data,valid_data,test_data
    metrics = [MAE(),NDCG1(),NDCG2(),RECALL1(),RECALL2(),Precision1(),Precision2()]
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