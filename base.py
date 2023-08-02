""" base model structure from which all other inherit

provides all utilities for model fitting apart from model architecture
"""

import copy
import os

# import comet_ml
# from comet_ml import Experiment

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, r2_score, roc_auc_score, mean_squared_error
from torch import nn, optim
from torch.autograd import Variable
from tqdm import tqdm


#from ekg_scd.helpers.loss import SupervisedContrastiveLoss, SupConLoss
#from ray import tune
#from ray.tune import CLIReporter
#from ray.tune.schedulers import ASHAScheduler


def roc_auc_score_nan(y, p):
    nan_idx = pd.isnull(y) | pd.isnull(p)
    try:
        score = roc_auc_score(y[~nan_idx], p[~nan_idx])
    except:
        score = float("NaN")
    return score


def f1_score_nan(y, p):
    nan_idx = np.isnan(y)
    return f1_score(y[~nan_idx], p[~nan_idx])


def r2_score_nan(y, p):
    nan_idx = pd.isnull(y) | pd.isnull(p)
    try:
        score = r2_score(y[~nan_idx], p[~nan_idx])
    except:
        score = float("NaN")
    return score


class Model(nn.Module):
    """base model class w/ some helper functions for training/manipulating
    parameters, and saving
    """

    def __init__(self, **kwargs):
        super(Model, self).__init__()
        self.kwargs = kwargs
        self.fit_res = None

    def save(self, filename):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "kwargs": self.kwargs,
                "fit_res": self.fit_res,
                "model_class": type(self),
            },
            f=filename,
        )

    def fit(self, data):
        raise NotImplementedError

    def lossfun(self, data, target):
        raise NotImplementedError

    def init_params(self):
        for p in self.parameters():
            if p.requires_grad == True:
                p.data.uniform_(-0.05, 0.05)

    def fix_params(self):
        for p in self.parameters():
            p.requires_grad = False

    def free_params(self):
        for p in self.parameters():
            p.requires_grad = True

    def num_params(self):
        return np.sum([p.numel() for p in self.parameters()])


def load_model(fname):
    model_dict = torch.load(fname)
    mod = model_dict["model_class"](**model_dict["kwargs"])
    mod.load_state_dict(model_dict["state_dict"])
    mod.fit_res = model_dict["fit_res"]
    return mod


class MaskedBCELoss(nn.Module):
    """BCELoss that accounts for NaNs (given by mask)"""

    def __init__(self, reduction="mean"):
        super(MaskedBCELoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction=reduction)

    def forward(self, output, target, mask):
        """masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = mask.view(-1)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.bce_loss(ovalid, tvalid)


def isnan(x):
    return x != x


class NanBCEWithLogitsLoss(nn.Module):
    """BCELoss that accounts for NaNs (given by mask)"""

    def __init__(self, reduction="mean"):
        super(NanBCEWithLogitsLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, output, target):
        """masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = Variable(~isnan(tvec).data)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.bce_loss(ovalid, tvalid)


class NanMSELoss(nn.Module):
    """BCELoss that accounts for NaNs (given by mask)"""

    def __init__(self, reduction="mean"):
        super(NanMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction="mean")

    def forward(self, output, target):
        """masked binary cross entropy loss
        Args:
          - output: batch_size x D float tensor with values in [0, 1]
          - target: batch_size x D float tensor with values in {0, 1}
          - mask  : batch_size x D byte tensor with 1 = not nan (include in loss)
        """
        tvec = target.view(-1)
        ovec = output.view(-1)
        mvec = Variable(~isnan(tvec).data)

        # grab valid --- return bce loss
        tvalid = tvec.masked_select(mvec)
        ovalid = ovec.masked_select(mvec)
        return self.mse_loss(ovalid, tvalid)


##############################
# standard fitting procedure #
##############################


def fit_model(model, train_loader, val_loader, **kwargs):
    do_cuda = kwargs.get("do_cuda", torch.cuda.is_available())
    min_epochs = kwargs.get("min_epochs", 40)
    max_epochs = kwargs.get("max_epochs", 100)
    patience = kwargs.get("patience", 10)
    weight_decay = kwargs.get("weight_decay", 1e-5)
    learning_rate = kwargs.get("learning_rate", 1e-2)
    lr_reduce_interval = kwargs.get("lr_reduce_interval", 10)
    lr_sched_gamma = kwargs.get("lr_sched_gamma", 0.5)
    opt_type = kwargs.get("optimizer", "adam")
    log_interval = kwargs.get("log_interval", False)
    warm_start_path = kwargs.get("warm_start_path", None)
    warm_start_from_zero = kwargs.get("warm_start_from_zero", False)
    save_path = kwargs.get("save_path", None)

    print("-------------------")
    print("fitting model: ", kwargs)

    # experiment = Experiment(
    #     api_key='7smwpzl0FeZJcESqBITDniX7I',
    #     project_name='ecg-resnet',
    #     workspace='cardiac-twins',
    # )

    hyper_params = {
    "learning_rate": learning_rate,
    "max_epochs": max_epochs,
    }
    # experiment.log_parameters(hyper_params)

    if not os.path.exists(save_path):
        os.mkdir(save_path)


    # set up optimizer
    plist = list(filter(lambda p: p.requires_grad, model.parameters()))
    if opt_type == "adam":
        optimizer = optim.Adam(plist, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(plist, lr=learning_rate, weight_decay=weight_decay)

    if do_cuda:
        torch.cuda.set_device("cuda:0")
        model.cuda()

    model_dict = {}
    model_dict["train_loss"] = []
    model_dict["val_loss"] = []
    model_dict["best_loss"] = np.inf
    start_epoch = 0

    if warm_start_path:
        checkpoint_dict = torch.load(warm_start_path)

        if warm_start_from_zero:
            start_dict = model.state_dict()
            init_dict = {k: v for k, v in checkpoint_dict["best_state"].items() if k not in ["out.weight", "out.bias"]}
            start_dict.update(init_dict)
            model.load_state_dict(start_dict)

        else:
            # Start up from previous state and epoch
            model.load_state_dict(checkpoint_dict["stop_state"])
            optimizer.load_state_dict(checkpoint_dict["stop_optimizer"])
            start_epoch = checkpoint_dict["stop_epoch"] + 1

            # Establish performance stats
            model_dict = {
                key: checkpoint_dict[key]
                for key in ["best_epoch", "best_loss", "best_stat", "best_state", "train_loss", "val_loss"]
            }

    for epoch in range(start_epoch, max_epochs):

        tloss, tstat = run_epoch(
            epoch, model, train_loader, optimizer, do_cuda, only_compute_loss=False, log_interval=log_interval
        )
        vloss, vstat = run_epoch(
            epoch, model, val_loader, optimizer, do_cuda, only_compute_loss=True, log_interval=log_interval
        )

        print(f"Loss ({epoch}): {round(vloss,6)}, Stat: {[round(float(v),6) for v in vstat]}")

        # model_dict["train_loss"].append(tloss)
        # model_dict["val_loss"].append(vloss)

        # wandb_dict = {'train loss': tloss, 'val loss': vloss}

        # for i in range(len(tstat)):
        #     wandb_dict["train stat " + str(i)] = tstat[i]
        #     wandb_dict["val stat " + str(i)] = vstat[i]

        # experiment.log_metrics(wandb_dict, epoch=epoch)

        if vloss < model_dict["best_loss"]:
            print("  (updating best loss)")
            model_dict["best_loss"] = vloss
            model_dict["best_stat"] = vstat

            # Save new best parameters
            model_dict["best_epoch"] = epoch
            model_dict["best_state"] = copy.deepcopy(model.state_dict())

        warm_start_dict = {
            "stop_epoch": epoch,
            "stop_state": copy.deepcopy(model.state_dict()),
            "stop_optimizer": copy.deepcopy(optimizer.state_dict()),
        }

        if (epoch >= model_dict["best_epoch"] + patience and epoch >= min_epochs) or epoch == max_epochs:
            print(f"Failed to improve loss for {patience} epochs. Stopping at epoch {epoch}")
            break

        # Save every epoch so that we can warm_start from the last epoch
        torch.save({**model_dict, **warm_start_dict}, f=save_path + "/current.best.pth.tar")

        if epoch % lr_reduce_interval == 0 and epoch != 0:
            print("... reducing learning rate!")
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_sched_gamma

    torch.save(model_dict, f=save_path + "/model.best.pth.tar")
    os.remove(save_path + "/current.best.pth.tar")

    print(f"Training complete! Outputs and hyperparameters saved to {save_path}")


def run_epoch(epoch, model, data_loader, optimizer, do_cuda, only_compute_loss=False, log_interval=20):
    if only_compute_loss:
        model.eval()
    else:
        model.train()

    # iterate over batches
    total_loss = 0
    trues, preds = [], []
    print(f"Enumerating batches, (epoch {epoch})")
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = Variable(data), Variable(target)
        if do_cuda:
            data, target = data.cuda(), target.cuda()
            data, target = data.contiguous(), target.contiguous()

        # set up optimizer
        if not only_compute_loss:
            optimizer.zero_grad()
        
        # ensure right type of data and target
        data = data.float()
        target = target.float()

        loss, logitpreds = model.lossfun(data, target)

        # backprop
        if not only_compute_loss:
            loss.backward()
            optimizer.step()

        # track pred probs
        logitpreds[:, ~model.regress] = torch.sigmoid(logitpreds[:, ~model.regress])

        trues.append(target.data.cpu().numpy())
        preds.append(logitpreds.data.cpu().numpy())
        total_loss += loss.item()
        if (log_interval != False) and (batch_idx % log_interval == 0):
            print(
                "{pre} Epoch: {ep} [{cb}/{tb} ({frac:.0f}%)]\tLoss: {loss:.6f}".format(
                    pre="  Val" if only_compute_loss else "  Train",
                    ep=epoch,
                    cb=batch_idx * data_loader.batch_size,
                    tb=len(data_loader.dataset),
                    frac=100.0 * batch_idx / len(data_loader),
                    loss=total_loss / (batch_idx + 1),
                )
            )

        # To prevent memory errors
        del loss, data, target, logitpreds
        torch.cuda.empty_cache()

    total_loss /= len(data_loader)
    trues, preds = np.row_stack(trues), np.row_stack(preds)

    # Calculate relevant statistic
    
    # stats = [
    #     r2_score_nan(true, pred) if r == True else roc_auc_score_nan(true, pred)
    #     for true, pred, r in zip(trues.T, preds.T, model.regress)
    # ]

    stats = [
        mean_squared_error(true, pred) if r == True else roc_auc_score_nan(true, pred)
        for true, pred, r in zip(trues.T, preds.T, model.regress)
    ]

    return total_loss, stats

def fit_model_contrastive(model, train_loader, val_loader, **kwargs):
    do_cuda = kwargs.get("do_cuda", torch.cuda.is_available())
    min_epochs = kwargs.get("min_epochs", 40)
    max_epochs = kwargs.get("max_epochs", 100)
    regress = kwargs.get("regress", [False])
    patience = kwargs.get("patience", 10)
    temperature = kwargs.get('temperature', 0.07)
    weight_decay = kwargs.get("weight_decay", 1e-5)
    learning_rate = kwargs.get("learning_rate", 1e-2)
    learning_rate_contrastive = kwargs.get("learning_rate_contrastive", 1e-1)
    lr_reduce_interval = kwargs.get("lr_reduce_interval", 10)
    lr_sched_gamma = kwargs.get("lr_sched_gamma", 0.5)
    opt_type = kwargs.get("optimizer", "adam")
    log_interval = kwargs.get("log_interval", False)
    warm_start_path = kwargs.get("warm_start_path", None)
    warm_start_from_zero = kwargs.get("warm_start_from_zero", False)
    save_path = kwargs.get("save_path", None)

    print("-------------------")
    print("fitting model: ", kwargs)

    # set up optimizer
    plist = list(filter(lambda p: p.requires_grad, model.parameters()))
    if opt_type == "adam":
        optimizer = optim.Adam(plist, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(plist, lr=learning_rate, weight_decay=weight_decay)

    model_dict = {}
    model_dict["train_loss"] = []
    model_dict["val_loss"] = []
    model_dict["best_loss"] = np.inf
    model_dict["best_epoch"] = 0
    start_epoch = 0

    loss = SupConLoss(temperature=temperature)

    if do_cuda:
        torch.cuda.set_device("cuda:0")
        model.cuda()


    if warm_start_path:
        checkpoint_dict = torch.load(warm_start_path)

        if warm_start_from_zero:
            start_dict = model.state_dict()
            init_dict = {k: v for k, v in checkpoint_dict["best_state"].items() if k not in ["out.weight", "out.bias"]}
            start_dict.update(init_dict)
            model.load_state_dict(start_dict)

        else:
            # Start up from previous state and epoch
            model.load_state_dict(checkpoint_dict["stop_state"])
            optimizer.load_state_dict(checkpoint_dict["stop_optimizer"])
            start_epoch = checkpoint_dict["stop_epoch"] + 1

            # Establish performance stats
            model_dict = {
                key: checkpoint_dict[key]
                for key in ["best_epoch", "best_loss", "best_stat", "best_state", "train_loss", "val_loss"]
            }

    for epoch in range(start_epoch, max_epochs):

        optimizer = optim.Adam(model.parameters(), lr=learning_rate_contrastive, weight_decay=weight_decay)

        tloss = run_epoch_contrastive(
            epoch, model, train_loader, optimizer, do_cuda, only_compute_loss=False, log_interval=log_interval, loss_func=loss
        )
         # Load checkpoint.
        #print("==> Resuming from checkpoint..")
        #assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        #checkpoint = torch.load("./checkpoint/ckpt_contrastive.pth")
        #model.load_state_dict(checkpoint["net"])

        model.fix_params_resnet()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        tloss, tstat = run_epoch_ce(
            epoch, model, train_loader, optimizer, do_cuda, only_compute_loss=False, log_interval=log_interval, regress=regress #optimizing the last layer
        )

        vloss, vstat = run_epoch_ce(
            epoch, model, val_loader, optimizer, do_cuda, only_compute_loss=True, log_interval=log_interval, regress=regress
        )

        print(f"Loss ({epoch}): {round(vloss,6)}, Contr Loss: {round(tloss,6)}, Stat: {[round(float(v),6) for v in vstat]}")

        model_dict["train_loss"].append(tloss)
        model_dict["val_loss"].append(vloss)

        if vloss < model_dict["best_loss"]:
            print("  (updating best loss)")
            model_dict["best_loss"] = vloss
            model_dict["best_stat"] = vstat

            # Save new best parameters
            model_dict["best_epoch"] = epoch
            model_dict["best_state"] = copy.deepcopy(model.state_dict())

        warm_start_dict = {
            "stop_epoch": epoch,
            "stop_state": copy.deepcopy(model.state_dict()),
            "stop_optimizer": copy.deepcopy(optimizer.state_dict()),
        }

        if (epoch >= model_dict["best_epoch"] + patience and epoch >= min_epochs) or epoch == max_epochs:
            print(f"Failed to improve loss for {patience} epochs. Stopping at epoch {epoch}")
            break

        # Save every epoch so that we can warm_start from the last epoch
        torch.save({**model_dict, **warm_start_dict}, f=save_path + "/current.best.pth.tar")

        if epoch % lr_reduce_interval == 0 and epoch != 0:
            print("... reducing learning rate!")
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_sched_gamma

        model.free_params_resnet()

    torch.save(model_dict, f=save_path + "/model.best.pth.tar")
    os.remove(save_path + "/current.best.pth.tar")

    print(f"Training complete! Outputs and hyperparameters saved to {save_path}")


def run_epoch_contrastive(epoch, model, data_loader, optimizer, do_cuda, loss_func, only_compute_loss=False, log_interval=20):
    if only_compute_loss:
        model.eval()
    else:
        model.train()

    # iterate over batches
    #print(loss_func)
    total_loss = 0
    trues, preds = [], []
    print(f"Enumerating batches, (epoch {epoch})")
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = Variable(data), Variable(target)
        if do_cuda:
            data, target = data.cuda(), target.cuda()
            data, target = data.contiguous(), target.contiguous()
            loss_func = loss_func.cuda()

        # set up optimizer
        if not only_compute_loss:
            optimizer.zero_grad()

        # push data through model (make sure the recon batch batches data)
        # TODO update to contrastive loss here
        projections = model.forward_contrastive(data)
        #print(projections.shape)
        #print(target.shape)
        #print('Nan in target?')
        #print(torch.isnan(target).any())
        #print('Nan in projection?')
        #print(torch.isnan(projections).any())

        loss = loss_func(projections.unsqueeze(1),target.squeeze(1))
        #print(loss)
        #print(loss.item())

        #loss, logitpreds = model.lossfun(data, target)

        # backprop
        if not only_compute_loss:
            loss.backward()
            optimizer.step()

        # track pred probs
        #logitpreds[:, ~model.regress] = torch.sigmoid(logitpreds[:, ~model.regress])

        #trues.append(target.data.cpu().numpy())
        #preds.append(logitpreds.data.cpu().numpy())
        total_loss += loss.item()
        if (log_interval != False) and (batch_idx % log_interval == 0):
            print(
                "{pre} Epoch: {ep} [{cb}/{tb} ({frac:.0f}%)]\tLoss: {loss:.6f}".format(
                    pre="  Val" if only_compute_loss else "  Train",
                    ep=epoch,
                    cb=batch_idx * data_loader.batch_size,
                    tb=len(data_loader.dataset),
                    frac=100.0 * batch_idx / len(data_loader),
                    loss=total_loss / (batch_idx + 1),
                )
            )

        # To prevent memory errors
        del loss, data, target #, logitpreds
        torch.cuda.empty_cache()

    total_loss /= len(data_loader)
    #trues, preds = np.row_stack(trues), np.row_stack(preds)

    # Calculate relevant statistic
    #stats = [
        #r2_score_nan(true, pred) if r == True else roc_auc_score_nan(true, pred)
        #for true, pred, r in zip(trues.T, preds.T, model.regress)
    #]

    return total_loss #, stats

def run_epoch_ce(epoch, model, data_loader, optimizer, do_cuda, regress, loss_func = NanBCEWithLogitsLoss(), only_compute_loss=False, log_interval=20):
    if only_compute_loss:
        model.eval()
    else:
        model.train()

    # iterate over batches
    total_loss = 0
    trues, preds = [], []
    print(f"Enumerating batches, (epoch {epoch})")
    for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
        data, target = Variable(data), Variable(target)
        if do_cuda:
            data, target = data.cuda(), target.cuda()
            data, target = data.contiguous(), target.contiguous()
            loss_func = loss_func.cuda()

        # set up optimizer
        if not only_compute_loss:
            optimizer.zero_grad()

        # push data through model (make sure the recon batch batches data)
        # TODO update to contrastive loss here
        predictions = model(data)
        loss = loss_func(predictions,target.squeeze(1))


        #loss, logitpreds = model.lossfun(data, target)

        # backprop
        if not only_compute_loss:
            loss.backward()
            optimizer.step()

        # track pred probs
        #logitpreds[:, ~model.regress] = torch.sigmoid(logitpreds[:, ~model.regress])
        predictions[:, ~np.array(regress)] = torch.sigmoid(predictions[:, ~np.array(regress)])

        trues.append(target.data.cpu().numpy())
        preds.append(predictions.data.cpu().numpy())
        total_loss += loss.item()
        if (log_interval != False) and (batch_idx % log_interval == 0):
            print(
                "{pre} Epoch: {ep} [{cb}/{tb} ({frac:.0f}%)]\tLoss: {loss:.6f}".format(
                    pre="  Val" if only_compute_loss else "  Train",
                    ep=epoch,
                    cb=batch_idx * data_loader.batch_size,
                    tb=len(data_loader.dataset),
                    frac=100.0 * batch_idx / len(data_loader),
                    loss=total_loss / (batch_idx + 1),
                )
            )

        # To prevent memory errors
        del loss, data, target #, logitpreds
        torch.cuda.empty_cache()

    total_loss /= len(data_loader)
    trues, preds = np.row_stack(trues), np.row_stack(preds)

    # Calculate relevant statistic
    stats = [
        r2_score_nan(true, pred) if r == True else roc_auc_score_nan(true, pred)
        for true, pred, r in zip(trues.T, preds.T, np.array(regress))
    ]

    return total_loss , stats

def fit_model_contrastive_tune(model, train_loader, val_loader, **kwargs):
    do_cuda = kwargs.get("do_cuda", torch.cuda.is_available())
    min_epochs = kwargs.get("min_epochs", 40)
    max_epochs = kwargs.get("max_epochs", 100)
    regress = kwargs.get("regress", [False])
    patience = kwargs.get("patience", 10)
    temperature = kwargs.get('temperature', 0.07)
    weight_decay = kwargs.get("weight_decay", 1e-5)
    learning_rate = kwargs.get("learning_rate", 1e-2)
    learning_rate_contrastive = kwargs.get("learning_rate_contrastive", 1e-1)
    lr_reduce_interval = kwargs.get("lr_reduce_interval", 10)
    lr_sched_gamma = kwargs.get("lr_sched_gamma", 0.5)
    opt_type = kwargs.get("optimizer", "adam")
    log_interval = kwargs.get("log_interval", False)
    warm_start_path = kwargs.get("warm_start_path", None)
    warm_start_from_zero = kwargs.get("warm_start_from_zero", False)
    save_path = kwargs.get("save_path", None)

    print("-------------------")
    print("fitting model: ", kwargs)

    # set up optimizer
    plist = list(filter(lambda p: p.requires_grad, model.parameters()))
    if opt_type == "adam":
        optimizer = optim.Adam(plist, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(plist, lr=learning_rate, weight_decay=weight_decay)

    model_dict = {}
    model_dict["train_loss"] = []
    model_dict["val_loss"] = []
    model_dict["best_loss"] = np.inf
    model_dict["best_epoch"] = 0
    start_epoch = 0

    loss = SupConLoss(temperature=temperature)

    if do_cuda:
        torch.cuda.set_device("cuda:0")
        model.cuda()


    if warm_start_path:
        checkpoint_dict = torch.load(warm_start_path)

        if warm_start_from_zero:
            start_dict = model.state_dict()
            init_dict = {k: v for k, v in checkpoint_dict["best_state"].items() if k not in ["out.weight", "out.bias"]}
            start_dict.update(init_dict)
            model.load_state_dict(start_dict)

        else:
            # Start up from previous state and epoch
            model.load_state_dict(checkpoint_dict["stop_state"])
            optimizer.load_state_dict(checkpoint_dict["stop_optimizer"])
            start_epoch = checkpoint_dict["stop_epoch"] + 1

            # Establish performance stats
            model_dict = {
                key: checkpoint_dict[key]
                for key in ["best_epoch", "best_loss", "best_stat", "best_state", "train_loss", "val_loss"]
            }

    for epoch in range(start_epoch, max_epochs):

        optimizer = optim.Adam(model.parameters(), lr=learning_rate_contrastive, weight_decay=weight_decay)

        tloss = run_epoch_contrastive(
            epoch, model, train_loader, optimizer, do_cuda, only_compute_loss=False, log_interval=log_interval, loss_func=loss
        )
         # Load checkpoint.
        #print("==> Resuming from checkpoint..")
        #assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
        #checkpoint = torch.load("./checkpoint/ckpt_contrastive.pth")
        #model.load_state_dict(checkpoint["net"])

        model.fix_params_resnet()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        tloss, tstat = run_epoch_ce(
            epoch, model, train_loader, optimizer, do_cuda, only_compute_loss=False, log_interval=log_interval, regress=regress #optimizing the last layer
        )

        vloss, vstat = run_epoch_ce(
            epoch, model, val_loader, optimizer, do_cuda, only_compute_loss=True, log_interval=log_interval, regress=regress
        )

        print(f"Loss ({epoch}): {round(vloss,6)}, Contr Loss: {round(tloss,6)}, Stat: {[round(float(v),6) for v in vstat]}")

        model_dict["train_loss"].append(tloss)
        model_dict["val_loss"].append(vloss)

        if vloss < model_dict["best_loss"]:
            print("  (updating best loss)")
            model_dict["best_loss"] = vloss
            model_dict["best_stat"] = vstat

            # Save new best parameters
            model_dict["best_epoch"] = epoch
            model_dict["best_state"] = copy.deepcopy(model.state_dict())

        warm_start_dict = {
            "stop_epoch": epoch,
            "stop_state": copy.deepcopy(model.state_dict()),
            "stop_optimizer": copy.deepcopy(optimizer.state_dict()),
        }

        if (epoch >= model_dict["best_epoch"] + patience and epoch >= min_epochs) or epoch == max_epochs:
            print(f"Failed to improve loss for {patience} epochs. Stopping at epoch {epoch}")
            break

        # Save every epoch so that we can warm_start from the last epoch
        torch.save({**model_dict, **warm_start_dict}, f=save_path + "/current.best.pth.tar")

        if epoch % lr_reduce_interval == 0 and epoch != 0:
            print("... reducing learning rate!")
            for param_group in optimizer.param_groups:
                param_group["lr"] *= lr_sched_gamma

        model.free_params_resnet()

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=vloss, accuracy=vstat[0])


    torch.save(model_dict, f=save_path + "/model.best.pth.tar")
    os.remove(save_path + "/current.best.pth.tar")

    print(f"Training complete! Outputs and hyperparameters saved to {save_path}")
