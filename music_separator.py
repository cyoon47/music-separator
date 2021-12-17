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
        super(MusicSeparator, self).__init__()
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
        self.in_features = 4*2*1323648 # PRAY HARD
        self.classifier = nn.Linear(self.in_features, 50)

    
    def forward(self, audio: Tensor) -> Tensor:
        # Returns stacked tensor of separated waveforms
        #   shape (nb_samples, nb_targets, nb_channels, nb_timesteps)
        
        estimates = self.separator(audio)
        estimates = estimates.view((-1, self.in_features))
        pred_logits = self.classifier(estimates)

        return pred_logits

    def train(self, mode=True):
        r"""Sets the module in training mode."""      
        self.training = mode
        for module in self.children():
            module.train(mode)
        self.separator.freeze()
        return self

    def eval(self):
        r"""Sets the module in evaluation mode."""
        return self.train(False)
