import io

import numpy as np
import pickle
import sys

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from earlystopping import EarlyStopping


class TrainerBase:
    def __init__(self, epochs, evaluate_steps=1,
                 valid_on_train_set = False,
                 valid_on_test_set = True,
                 verbose=True):
        self.epochs = epochs
        self.verbose = verbose
        self.evaluate_steps = evaluate_steps
        self.valid_on_train_set = valid_on_train_set
        self.valid_on_test_set = valid_on_test_set

    def save_state(self, filepath,epch,early_stopping):
        trainer_filepath = filepath + '.last_trainner_state.pkl'
        state={'epcho': epch,
        'earl_stopping':early_stopping
        }
        with open(trainer_filepath,'wb') as fout:
            pickle.dump(state, fout)

    def load_state(self, filepath,epch,early_stopping):
        state = {'epcho': epch,
                 'earl_stopping': early_stopping
                 }
        trainer_filepath = filepath + '.last_trainner_state.pkl'
        try :
            with open(trainer_filepath,'rb') as fin:
                state = pickle.load( fin)
        except:
            pass
        return state['epcho'],state['earl_stopping']


    def _train_epoch(self, model, train_loader, epch):

        iter_data = (
            tqdm(
                train_loader,
                total=len(train_loader),
                ncols=100,
                desc=f"Train {epch}:>5"
            )
            if self.verbose
            else train_loader
        )
        headers = None
        data = []
        for batch_data in iter_data:
            # batch是train_loader的一个batch
            result = model.opt_one_batch(batch_data)
            if headers is None :
                headers = result.keys()
            data.append(list(result.values()))
        # if len(iter_data) > 1:
        data = np.mean(np.array(data), axis=0)
        return dict(zip(headers,data))

    def _eval_data(self, model, dataloader, valid_fun):

        return model.eval_data(dataloader, valid_fun)


    def create_EarlyStopping(self, model):

        patience = 7
        delta = 0
        trace_func = print
        if hasattr(model, 'checkpoint_path'):
            checkpoint = model.checkpoint_path
        else:
            checkpoint = 'checkpoint'
        if hasattr(model, 'es_patience'):
            patience = model.es_patience
        if hasattr(model, 'es_delta'):
            delta = model.es_delta

        return EarlyStopping(patience, self.verbose, delta, trace_func, checkpoint)

    def train(self, model, train_loader, valid_loader,test_loader, valid_func, loger=sys.stdout):


        def printf(key, value, epch):
            if isinstance(loger, SummaryWriter):
                loger.add_scalar(key, value, global_step=epch)
            elif loger == sys.stdout:
                line = f"{key}={value}\t\t  epcho={epch}"
                loger.writelines(line)

        metric_name = str(valid_func)
        early_stoping = self.create_EarlyStopping(model)
        epch_start=0

        for epch in range(epch_start+1, self.epochs + 1):
            if early_stoping.early_stop: break
            results = self._train_epoch(model, train_loader, epch)
            if self.verbose:
                for key, value in results.items():
                    printf(key, value, epch)
            if self.evaluate_steps <=0 :
                continue

            if (epch - 1) % self.evaluate_steps == 0:
                if self.valid_on_train_set:

                    train_score = self._eval_data(model, train_loader, valid_func)
                    printf(f"{metric_name}@train set", train_score, epch)
                if self.valid_on_test_set and test_loader is not None:

                    test_score = self._eval_data(model, test_loader, valid_func)
                    printf(f"{metric_name}@test set", test_score, epch)

                if valid_loader is not None:
                    val_score = self._eval_data(model, valid_loader, valid_func)
                    printf(f"{metric_name}@valid set", val_score, epch)
                elif self.valid_on_train_set:
                    val_score=train_score
                else:
                    val_score = -results['loss']
            if valid_loader != None and  hasattr(valid_func, 'bigger') and valid_func.bigger == False:
                val_score = -val_score  # 分数越小越好，则取反进行判断

            early_stoping(val_score, model, epch)





        if self.evaluate_steps > 0:
            early_stoping.get_best(model)
