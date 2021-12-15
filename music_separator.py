import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from openunmix import utils

class MusicSeparator(nn.Module):

    def __init__(self,
        model_str_or_path="umxhq",
        targets=None,
        niter=1,
        residual=False,
        wiener_win_len=300,
        device=None
    ):
        self.separator = utils.load_separator(
            model_str_or_path=model_str_or_path,
            targets=targets,
            niter=niter,
            residual=residual,
            wiener_win_len=wiener_win_len,
            device=device,
            pretrained=True
        )
        self.separator.freeze()
        
        # TODO: figure out what is the input tensor size
        in_features = None
        self.classifier = nn.Linear(in_features, 100)

    
    def forward(self, audio: Tensor) -> Tensor:
        # Returns stacked tensor of separated waveforms
        #   shape (nb_samples, nb_targets, nb_channels, nb_timesteps)
        
        estimates = self.separator(audio)
        pred_logits = self.classifier(estimates)

        return pred_logits