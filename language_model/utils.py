import matplotlib.pyplot as plt
import requests
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM
import numpy as np
import re
import inspect
from torch import optim
import collections


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
                    if isinstance(v, dict):
                        self[k] = Config(v)

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v
                if isinstance(v, dict):
                    self[k] = Config(v)

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Config, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Config, self).__delitem__(key)
        del self.__dict__[key]


class Monitor():
    def __init__(self, path="/data/medioli/train_log_baseline/", train_file="train_log.txt", pred_file="pred_log.txt",
                 elements=["step", "train", "pred"], reg_loss_queue_len=1000):
        print("Monitor - Init...")
        self.path = path
        self.train_file = train_file
        t_file = open(path + self.train_file, "w")
        t_file.write("step\ttrain_loss\treg_loss\ttotal_loss\n")
        t_file.close()
        self.reg_loss_queue = collections.deque(maxlen=reg_loss_queue_len)
        self.pred_file = pred_file
        t_pred_file = open(path + self.pred_file, "w")
        t_pred_file.write("step\tpred_loss\treg_loss\n")
        t_pred_file.close()
        self.step = 0
        print("Monitor - End Init")

    def log(self, loss, element, reg_loss=None, without=None):
        if reg_loss:
            reg_loss = reg_loss.item()
        # if without:
        #     reg_loss = without.item()
        loss = loss.item()
        monitor_file = open("train_monitor.txt", "w")
        monitor_file.write(str(self.step) + "\t" + str(without) + "\t" + str(reg_loss) + "\t" + str(loss) + "\n")
        monitor_file.close()
        if self.step % 100 == 0:
            if element == "train":
                train_file = open(self.path + self.train_file, "a")
                train_file.write(str(self.step) + "\t" + str(without) + "\t" + str(reg_loss) + "\t" + str(loss) + "\n")
                train_file.close()

                # self.reg_loss_queue.append(reg_loss)
                # TODO: scheduler to adapt lambda based on std and variance of reg. loss

            if element == "pred":
                pred_file = open(self.path + self.pred_file, "a")
                self.pred_file.write(str(self.step) + "\t" + str(loss) + "\n")
                pred_file.close()
        self.step += 1


def cuda_setup():
    if torch.cuda.is_available():

        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device


def get_batches(data, batch_size=32):
    return DataLoader(data, batch_size=batch_size)


def plot_pca(X, colors=None, n_components=3, element_to_plot=5000, path=None, epoch=0):
    if not colors:
        colors = np.ones(len(X))
    fig = plt.figure()
    pca = PCA(n_components=n_components)
    pca.fit(X)
    pca_components = pca.transform(X)
    if n_components == 2:
        plt.scatter(pca_components[:element_to_plot, 0],
                    pca_components[:element_to_plot, 1],
                    c=colors[:element_to_plot], s=1, marker='x')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pca_components[:element_to_plot, 0],
                   pca_components[:element_to_plot, 1],
                   pca_components[:element_to_plot, 2],
                   c=colors[:element_to_plot], s=1, marker='x')
    print("Saving png...")
    plt.savefig(path + str(epoch) + "e_pca.png")


def create_data_split(data, test_perc=0.1, val_perc=0.2):
    num_test = round(data.num_nodes * test_perc)
    num_val = round(data.num_nodes * val_perc)
    train_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(data.shape[0], dtype=torch.bool)
    perm = torch.randperm(data.shape[0])
    t = int(data.shape[0]) - (num_val + num_test)
    train_mask[perm[:t]] = True
    val_mask[perm[t:t + num_test]] = True
    test_mask[perm[t + num_test:]] = True
    return train_mask, val_mask, test_mask


def load_custom_model(path, checkpoint_fldr_and_bin, regularized=False, device='cuda'):
    state_dict = torch.load(path + checkpoint_fldr_and_bin, map_location=torch.device(device))
    keys = state_dict.keys()
    if regularized:
        for k in list(keys):
            if 'bert.' in k:
                state_dict[k[5:]] = state_dict[k]
                del state_dict[k]
    return BertForMaskedLM.from_pretrained(
        pretrained_model_name_or_path=path,
        state_dict=state_dict)


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getfullargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params