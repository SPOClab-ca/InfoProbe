import argparse
from collections import OrderedDict
import copy
import flair
import logging
import numpy as np 
import os, sys, time
import pandas as pd
import tempfile
from tqdm import tqdm 
import torch
import torch.nn as nn 
import transformers

from dataloader import DataLoader
from constants import spacy_pos_dict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class MLP(nn.Module):
    def __init__(self, args, dl):
        super().__init__()
        networks = OrderedDict()
        network_dim = [dl.emb_dim]
        for i in range(args.nlayers):
            network_dim.append(args.dim)
        network_dim.append(len(dl.pos_tags))
        for i in range(len(network_dim)-1):
            fc = nn.Linear(network_dim[i], network_dim[i+1])
            networks["fc_{}".format(i+1)] = fc
            networks["relu_{}".format(i+1)] = nn.ReLU()
        self.network = nn.Sequential(networks)
        self.to(device)

    def forward(self, x_tensor):
        return self.network(x_tensor)


def main(args):
    # Logging
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler = logging.FileHandler(args.log)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    ch = logging.StreamHandler() 
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(args)

    logger.info("Seed: {}".format(args.seed))
    torch.manual_seed(args.seed)

    
    
    # Training
    tr_dataloader = DataLoader("train", args)
    dev_dataloader = DataLoader("dev", args)
    probe = MLP(args, tr_dataloader)
    optim = torch.optim.Adam(
        probe.parameters(), 
        lr=args.lr, weight_decay=args.weight_decay)
    
    # Load checkpoint, or initialize training
    if os.path.exists(args.program_checkpoint):
        checkpoint = torch.load(args.program_checkpoint)
        best_dev_loss = checkpoint["best_dev_loss"]
        best_model_state_dict = checkpoint["best_model_state_dict"]
        epoch = checkpoint["epoch"]
        steps = checkpoint["steps"]
        probe.load_state_dict(checkpoint["probe"])
        optim.load_state_dict(checkpoint["optim"])
        logger.info("Resume training from checkpoint at epoch {}".format(epoch))
    else:
        best_dev_loss = np.inf
        best_model_state_dict = None 
        epoch = 0
        steps = 0
    
    # Start training
    doc_iter = 0
    epoch_loss_buffer = []
    stopping_counter = 0
    while args.max_grad_step < 0 or steps < args.max_grad_step:
        # One document per step. Minibatch of size 1. 
        x_tensor, y_tensor = tr_dataloader.next()
        if x_tensor is None:  # This epoch finishes
            epoch += 1
            tr_dataloader.reset()
            epoch_tr_loss = np.mean(epoch_loss_buffer)
            epoch_loss_buffer = []
            valid_loss, val_acc = run_eval(probe, dev_dataloader)
            if epoch % 20 == 0:  # Around 30s per epoch.
                torch.save({
                    "probe": probe.state_dict(),
                    "probe_best": best_model_state_dict,
                    "best_dev_loss": best_dev_loss,
                    "optim": optim.state_dict(),
                    "epoch": epoch,
                    "steps": steps
                }, args.program_checkpoint)

            if valid_loss < best_dev_loss:
                stopping_counter = 0
                best_dev_loss = valid_loss
                best_model_state_dict = copy.deepcopy(probe.state_dict())
            else:
                stopping_counter += 1
                # Anneal learning rate
                for param_group in optim.param_groups:
                    param_group["lr"] *= args.lr_anneal

            logger.info("Epoch {}: tr|dev {:.4f} | {:.4f}, Val acc {:.2f}".format(epoch, epoch_tr_loss, valid_loss, val_acc))

            if stopping_counter == 4:
                logger.info("Loss not improving for 4 consecutive epochs. Training done.")
                break

        else:  # This epoch is not done; keep training
            y_pred = nn.LogSoftmax(dim=-1)(probe(x_tensor))
            loss = nn.CrossEntropyLoss()(y_pred, y_tensor)
            loss.backward()
            optim.step()
            epoch_loss_buffer.append(loss.item())
            steps += 1
                
            if steps == args.max_grad_step:
                epoch_loss = np.mean(epoch_loss_buffer)
                valid_loss, val_acc = run_eval(probe, dev_dataloader)
                logger.info("Steps reached {}. Early stop. Valid loss {:.4f} Acc {:.2f}".format(steps, valid_loss, val_acc))
    
    test_dataloader = DataLoader("test", args)
    if best_model_state_dict is not None:  # This is None before the first epoch is done (early stopping)
        probe.load_state_dict(best_model_state_dict)
    test_loss, test_acc = run_eval(probe, test_dataloader)
    logger.info("Total steps: {}".format(steps))
    logger.info("Test: {:.4f} Acc {:.2f}".format(test_loss, test_acc))

    exp_df = pd.DataFrame({
            "lm": [args.lm], "lang": [args.lang], "task": [args.task],
            "layer": [args.nlayers], "dim": [args.dim],
            "batch_size": [args.batch_size],
            "init_lr*1e6": [args.lr*1e6], 
            "weight_decay": [args.weight_decay],
            "lr_anneal": [args.lr_anneal],
            "max_grad_step": args.max_grad_step,
            "train_steps": steps,
            "seed": [args.seed],
            "devloss": [best_dev_loss],
            "testloss": [test_loss],
            "acc": [test_acc]
        })

    # Bookkeeping
    if not os.path.exists(args.bookkeep_df):
        bk_df = None 
    else:
        bk_df = pd.read_csv(args.bookkeep_df)
    bk_df = pd.concat([bk_df, exp_df], sort=False)
    bk_df.to_csv(args.bookkeep_df, index=False)


def run_eval(probe, dataloader):
    probe.eval()
    eval_losses = []
    n_total, n_correct = 0, 0
    while dataloader.has_next():
        x_tensor, y_tensor = dataloader.next()
        if x_tensor is None:
            break
        y_pred = nn.LogSoftmax(dim=-1)(probe(x_tensor))
        loss = nn.CrossEntropyLoss()(y_pred, y_tensor)
        eval_losses.append(loss.item())

        n_total += len(y_tensor)
        predictions, predint = y_pred.max(dim=-1)
        n_correct += (predint == y_tensor).sum().item()
    dataloader.reset()
    probe.train()
    acc = n_correct / n_total * 100 
    return np.mean(eval_losses), acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lm", type=str, choices=["bertmulti", "fasttext", "glove"], default="bertmulti")
    parser.add_argument("--lang", type=str, choices=["en", "es", "fr"], default="en")
    parser.add_argument("--task", type=str, choices=["probe", "ctarget", "crep"], default="probe")
    parser.add_argument("--seed", type=int, default=73)

    parser.add_argument("--nlayers", type=int, default=1, 
            help="Num layers for probe")
    parser.add_argument("--dim", type=int, default=100, 
            help="Dimension for probe")
    parser.add_argument("--batch_size", type=int, default=32,
            help="Batch size for training. For valid / test the batch size is always 1")
    parser.add_argument("--lr", type=float, default=3e-4,
            help="Learning rate for Adam optimizer")
    parser.add_argument("--lr_anneal", type=float, default=1.0,
            help="Annealing for learning rate per epoch")
    parser.add_argument("--weight_decay", type=float, default=0,
            help="Weight decay for Adam optimizer")
    parser.add_argument("--max_grad_step", type=int, default=-1)
    parser.add_argument("--log", type=str, default="test_log.out")

    parser.add_argument("--bookkeep_df", type=str, default="bookkeep_df.csv")

    parser.add_argument("--program_checkpoint", type=str, default="",
            help="Only needed for V cluster.")

    args = parser.parse_args()

    main(args)
