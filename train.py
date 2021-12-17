import argparse
import torch
import time
from pathlib import Path
import tqdm
import json
import sklearn.preprocessing
import numpy as np
import random
from git import Repo
import os
import copy
import torchaudio

import data
from music_separator import MusicSeparator
from openunmix import utils
# from openunmix import transforms

tqdm.monitor_interval = 0



def main():
    parser = argparse.ArgumentParser(description="MusicSeparator Trainer")

    parser.add_argument("--train_path", type=str, help="path of train dataset")
    parser.add_argument("--valid_path", type=str, help="path of valid dataset")
    parser.add_argument("--model_path", type=str, default="", help="Name or path of pretrained model to fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate, defaults to 1e-3")
    parser.add_argument(
        "--patience",
        type=int,
        default=140,
        help="maximum number of train epochs (default: 140)",
    )
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=80,
        help="lr decay patience for plateau scheduler",
    )
    parser.add_argument(
        "--lr-decay-gamma",
        type=float,
        default=0.3,
        help="gamma of learning rate scheduler decay",
    )
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="weight decay")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="less verbose during training",
    )
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        default=False,
        help="use cuda",
    )
    parser.add_argument("--nb_workers", type=int, default=0)
    parser.add_argument("--audio_backend", type=str, default="sox_io")


    args, _ = parser.parse_known_args()
    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_dataset, valid_dataset = data.load_datasets(args.train_path, args.valid_path)

    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=1, **dataloader_kwargs)

    model = MusicSeparator()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_gamma,
        patience=args.lr_decay_patience,
        cooldown=10,
    )

    es = utils.EarlyStopping(patience=args.patience)

    if args.model_path:
        model_path = Path(args.model_path).expanduser()
        with open(Path(model_path, args.target + ".json"), "r") as stream:
            results = json.load(stream)
        
        target_model_path = Path(model_path, "model.chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

        t = tqdm.trange(
            results["epochs_trained"],
            results["epochs_trained"] + args.epochs + 1,
            disable=args.quiet,
        )

        train_losses = results["train_loss_history"]
        valid_losses = results["valid_loss_history"]
        train_times = results["train_time_history"]
        best_epoch = results["best_epoch"]
        es.best = results["best_loss"]
        es.num_bad_epochs = results["num_bad_epochs"]
    else:
        t = tqdm.trange(1, args.epochs + 1, disable=args.quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()

        losses = utils.AverageMeter()
        model.train()
        pbar = tqdm.tqdm(train_sampler, disable=args.quiet)
        criterion = torch.nn.CrossEntropyLoss()

        ##Train##
        for x, y in pbar:
            pbar.set_description("Training batch")

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_out = model(x)
            print(y_out.shape, y.shape) 
            loss = criterion(y_out, y)
            
            loss.backward()
            optimizer.step()
            losses.update(loss.item(), y.size(1))
        
        train_loss = losses.avg
        
        ##VALID##
        losses = utils.AverageMeter()
        model.eval()
        with torch.no_grad():
            for x, y in valid_sampler:
                x, y = x.to(device), y.to(device)
                y_out = model(x)
                loss = criterion(y_out, y)
                losses.update(loss.item(), y.size(1))
        
        valid_loss = losses.avg

        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(train_loss=train_loss, val_loss=valid_loss)

        stop = es.stop(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch


        state = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_loss": es.best,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, Path(model_path, "model.chkpnt"))
        if valid_loss == es.best:
            torch.save(state["state_dict"], os.path.join(model_path, "model.pth"))

        #save params
        params = {
            'epochs_trained': epoch,
            'args': vars(args),
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
        }

        with open(Path(target_model_path, args.target + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))
        train_times.append(time.time() - end)

        if stop:
            print("Early Stopping")
            break

if __name__ == "__main__":
    main()
