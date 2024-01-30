
import io
import logging

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    '''
    Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            tmpFile (file): file for the checkpoint to be saved to.
                            Default: 'io.BytesIO()'
            trace_func (function): trace print function.
                            Default: print
    '''
    def __init__(self, patience=7, verbose=False, delta=0,  trace_func=print, tmpFile=io.BytesIO()):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score = float("-inf")
        self.delta = delta
        self.tmpFile = tmpFile
        self.trace_func = trace_func  #<class 'builtin_function_or_method'>

    def __call__(self, score, model,epch):


        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}, at Epcho {epch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def get_best(self, model):
        if isinstance(self.tmpFile,io.IOBase):
            self.tmpFile.seek(0)
        return model.load(self.tmpFile)

    def save_checkpoint(self, val_loss, model):
        '''Saves models when validation loss decrease.'''
        if self.verbose:
            logging.info(f'validate metric increased {self.val_score:.6f} --> {val_loss:.6f}).  Saving models ...')
        if isinstance(self.tmpFile, io.IOBase):
            self.tmpFile.seek(0)
        model.save(self.tmpFile)
        self.val_score = val_loss