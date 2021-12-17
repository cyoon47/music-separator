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

def F1_score(prob, label):
    prob = prob.bool()
    label = label.bool()
    epsilon = 1e-7
    TP = (prob & label).sum().float()
    TN = ((~prob) & (~label)).sum().float()
    FP = (prob & (~label)).sum().float()
    FN = ((~prob) & label).sum().float()
    #accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = torch.mean(TP / (TP + FP + epsilon))
    recall = torch.mean(TP / (TP + FN + epsilon))
    F2 = 2 * precision * recall / (precision + recall + epsilon)
    return precision, recall, F2

def main():
    parser = argparse.ArgumentParser(description="MusicSeparator Tester")

    parser.add_argument("--test_path", type=str, help="path of test dataset")
    parser.add_argument("--model_path", type=str, default="", required=True, help="Name or path of pretrained model")
    parser.add_argument(
        "--seed", type=int, default=42, metavar="S", help="random seed (default: 42)"
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
    torchaudio.USE_SOUNDFILE_LEGACY_INTERFACE = False
    torchaudio.set_audio_backend(args.audio_backend)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    print("Using GPU:", use_cuda)
    dataloader_kwargs = {"num_workers": args.nb_workers, "pin_memory": True} if use_cuda else {}

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    test_dataset = data.load_test_dataset(args.test_path)

    test_sampler = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True, **dataloader_kwargs
    )

    model = MusicSeparator()
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    if args.model_path:
        model_path = Path(args.model_path).expanduser()
        
        target_model_path = Path(model_path, "model.chkpnt")
        checkpoint = torch.load(target_model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

    model.eval()
    num_correct = 0
    percentage_positive = 0
    p, r, f1 = 0, 0, 0
    with torch.no_grad():
        for x, y in tqdm.tqdm(test_sampler):
            x, y = x.to(device), y.to(device)
            y_out = model(x)
            
            y_out = (y_out>0.5).float()

            # num_correct += torch.sum(torch.sum(y_out == y, 1) / y_out.shape[1]).item() / y_out.shape[0]
            num_correct += torch.all(y_out == y).int().item()
            p_c,r_c,f1_c = F1_score(y_out, y)
            p += p_c
            r += r_c
            f1 += f1_c
    
    print("Test accuracy:", num_correct / len(test_dataset))
    print("Precision: %.2f" % (p*100 / len(test_dataset)))
    print("Recall: %.2f" % (r*100 / len(test_dataset)))
    print("F1: %.2f" % (f1*100 / len(test_dataset)))


if __name__ == "__main__":
    main()
