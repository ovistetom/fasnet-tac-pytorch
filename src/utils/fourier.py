import torch
import torch.nn as nn
import torchaudio


class STFT(nn.Module):
    def __init__(
            self,
            win_size: int = 1024,
            hop_size: int = 512,
            win_func: str = 'hann',
            complex_output: bool = False,
            magnitude_only: bool = False,
            device: str | torch.device = 'cpu',
            **kwargs,
    ):
        """Extended STFT module.

        Args:
            win_size (int): Window size for the STFT. Default: `1024`.
            hop_size (int): Hop size for the STFT. Default: `512`.
            win_func (str): Window function to use. Default: `'hann'`.
            complex_output (bool): Whether to return the complex-valued STFT or stacked real & imaginary parts. Default: `False`.
            magnitude_only (bool): Whether to return the magnitude spectrogram. Default: `False`.
            device (str, torch.device): Device to use. Default: `'cpu'`.
            **kwargs: Additional keyword arguments to pass to the Spectrogram transform.
        """
        super().__init__()
        self.complex_output = complex_output
        self.magnitude_only = magnitude_only
        self.device = device
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft = win_size,
            win_length = win_size,
            hop_length = hop_size,
            window_fn = self._map_window_fn(win_func.lower()),
            power = 1 if magnitude_only else None,
            normalized = False,
            **kwargs,
        )

    def forward(self, waveform):
        """
        Args:
            waveform (torch.Tensor): Input waveform tensor.
                Tensor of size `[B, M, L]`.
        Returns:
            torch.Tensor: STFT of the input waveform.
                Tensor of size `[B, F, T]` if `complex_output` or `magnitude_only` is `True`, else `[B, F, T, 2]`.
        """
        if self.complex_output or self.magnitude_only:
            return self.stft(waveform)
        else:
            return torch.view_as_real(self.stft(waveform))

    def _map_window_fn(self, window_name):
        """Map window name to PyTorch window function to correctly handle device."""
        window_func = {
            'hann': lambda *args: torch.hann_window(*args).to(self.device),
            'sqrt_hann': lambda *args: torch.sqrt(torch.hann_window(*args).to(self.device)),
            'hamming': lambda *args: torch.hamming_window(*args).to(self.device),
            'bartlett': lambda *args: torch.bartlett_window(*args).to(self.device),
            'blackman': lambda *args: torch.blackman_window(*args).to(self.device),
            'kaiser': lambda *args: torch.kaiser_window(*args).to(self.device),
        }
        assert window_name in window_func, f"Unknown window function: '{window_name}'."
        return window_func[window_name]
    

class InverseSTFT(nn.Module):
    def __init__(
            self,
            win_size: int = 1024,
            hop_size: int = 512,
            win_func: str = 'hann',
            complex_input: bool = False,
            device: str | torch.device = 'cpu',
            **kwargs,
    ):
        """Extended inverse STFT module.

        Args:
            win_size (int): Window size for the STFT. Defaults to `1024`.
            hop_size (int): Hop size for the STFT. Defaults to `512`.
            win_func (str): Window function to use. Defaults to `'hann'`.
            complex_input (bool, optional): Whether the input STFT is complex. Defaults to `False`.
            device (str): Device to use. Defaults to `'cpu'`.
            **kwargs (): Additional keyword arguments for InverseSpectrogram.
        """
        super().__init__()
        self.device = device
        self.complex_input = complex_input
        self.istft = torchaudio.transforms.InverseSpectrogram(
            n_fft = win_size,
            win_length = win_size,
            hop_length = hop_size,
            window_fn = self._map_window_fn(win_func.lower()),
            normalized = False,
            **kwargs,
        )

    def forward(self, specgram, length=None):
        """
        Args:
            specgram (torch.Tensor): Input spectrogram tensor.
                Tensor of size `[B, F, T]` if `complex_input` is `True`, else `[B, F, T, 2]`.
            length (int, optional): Length of the output waveform.
                Defaults to `None`.
        Returns:
            torch.Tensor: Inverse STFT of the input spectrogram.
                Tensor of size `[B, M, L]`.
        """        
        if self.complex_input:
            return self.istft(specgram, length=length)
        else:
            return self.istft(torch.view_as_complex(specgram), length=length)

    def _map_window_fn(self, window_name):
        """Map window name to PyTorch window function to correctly handle device."""
        window_func = {
            'hann': lambda *args: torch.hann_window(*args).to(self.device),
            'sqrt_hann': lambda *args: torch.sqrt(torch.hann_window(*args).to(self.device)),
            'hamming': lambda *args: torch.hamming_window(*args).to(self.device),
            'bartlett': lambda *args: torch.bartlett_window(*args).to(self.device),
            'blackman': lambda *args: torch.blackman_window(*args).to(self.device),
            'kaiser': lambda *args: torch.kaiser_window(*args).to(self.device),
        }
        assert window_name in window_func, f"Unknown window function: '{window_name}'."
        return window_func[window_name]
    

class FeaturesSTFT(nn.Module):
    def __init__(
            self,
            features: str = 'real+imag',
    ):
        """Module that computes spectral features from STFT.

        Args:
            features (str, optional): Spectral features to compute.
                Example: `'real+imag'`, `'abs+cos+sin'`. Default: `'real+imag'`.
        """        
        super().__init__()
        assert features in {'real+imag', 'abs+ang', 'log+cos+sin', 'abs+cos+sin'}, f"Unknown features: '{features}'."
        self.features = features

    def forward(self, specgram: torch.Tensor):
        """
        Args:
            specgram (torch.Tensor): Complex-valued spectrogram.
                Tensor of size `[B, M, F, T]`.
        Returns:
            torch.Tensor: Spectral features.
                Tensor of size `[B, M, F, T, X]` where `X` is the number of features.
        """
        if self.features == 'real+imag':
            return torch.view_as_real(specgram)
        elif self.features == 'abs+ang':
            specgram_abs = torch.abs(specgram)
            specgram_ang = torch.angle(specgram)
            return torch.stack([specgram_abs, specgram_ang], dim=-1)
        elif self.features == 'log+cos+sin':
            specgram_log = torch.log(torch.abs(specgram).pow(2) + 1e-8)
            specgram_ang = torch.angle(specgram)
            specgram_cos = torch.cos(specgram_ang)
            specgram_sin = torch.sin(specgram_ang)
            return torch.stack([specgram_log, specgram_cos, specgram_sin], dim=-1)
        else:
            specgram_abs = torch.abs(specgram)
            specgram_ang = torch.angle(specgram)
            specgram_cos = torch.cos(specgram_ang)
            specgram_sin = torch.sin(specgram_ang)
            return torch.stack([specgram_abs, specgram_cos, specgram_sin], dim=-1)