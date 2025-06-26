import torch
import torchmetrics
import os
import sys

import torchmetrics.audio
import torchmetrics.audio.pesq
import torchmetrics.audio.sdr
import torchmetrics.audio.stoi
sys.path.append(os.path.abspath(''))
import src.utils.fourier as fourier

Loss = torch.nn.modules.loss._Loss


################################################################################################################################
class MultiLoss(Loss):
    def __init__(
            self, 
            waveform_criterion: Loss | None = None,
            specgram_criterion: Loss | None = None,
            param_clean_criterion: Loss | None = None,
            param_noise_criterion: Loss | None = None,
            waveform_criterion_scale: float = 1.0,
            specgram_criterion_scale: float = 1.0,
            param_clean_criterion_scale: float = 1.0,
            param_noise_criterion_scale: float = 1.0,
    ):
        """Class to combine multiple losses computed on different input quantites."""
        super().__init__()
        self.waveform_criterion = waveform_criterion
        self.specgram_criterion = specgram_criterion
        self.param_clean_criterion = param_clean_criterion
        self.param_noise_criterion = param_noise_criterion
        self.waveform_criterion_scale = waveform_criterion_scale
        self.specgram_criterion_scale = specgram_criterion_scale
        self.param_clean_criterion_scale = param_clean_criterion_scale
        self.param_noise_criterion_scale = param_noise_criterion_scale

    def forward(
            self,
            waveform_estim=None,
            param_clean=None,
            param_noise=None,
            waveform_truth=None,
            waveform_clean=None,
            waveform_noise=None,            
    ):
        loss = 0.0
        if self.waveform_criterion is not None:
            loss += self.waveform_criterion_scale * self.waveform_criterion(waveform_estim, waveform_truth)
        if self.specgram_criterion is not None:
            loss += self.specgram_criterion_scale * self.specgram_criterion(waveform_estim, waveform_truth)
        if self.param_clean_criterion is not None:
            loss += self.param_clean_criterion_scale * self.param_clean_criterion(param_clean, waveform_clean)
        if self.param_noise_criterion is not None:
            loss += self.param_noise_criterion_scale * self.param_noise_criterion(param_noise, waveform_noise, waveform_clean)
        return loss


################################################################################################################################
class LossL1(Loss):
    def __init__(self, *args, **kwargs):
        """Criterion that measures the L1 loss between estimated and ground-truth waveforms."""
        super().__init__()
        self.loss = torch.nn.L1Loss(*args, **kwargs)
    def forward(self, waveform_estim: torch.Tensor, waveform_truth: torch.Tensor):
        return self.loss(input=waveform_estim, target=waveform_truth)
    

class LossMSE(Loss):
    def __init__(self, *args, **kwargs):
        """Criterion that measures the MSE loss between estimated and ground-truth waveforms."""
        super().__init__()
        self.loss = torch.nn.MSELoss(*args, **kwargs)
    def forward(self, waveform_estim: torch.Tensor, waveform_truth: torch.Tensor):
        return self.loss(input=waveform_estim, target=waveform_truth)


class LossSDR(Loss):
    def __init__(
            self,
            reduction: str = 'mean',
        ):
        """Criterion that measures the SDR loss between estimated and ground-truth waveforms."""
        super().__init__()
        self.reduction = reduction

    def forward(self, waveform_estim: torch.Tensor, waveform_truth: torch.Tensor):
        error = - 10 * torch.log10( waveform_truth.pow(2.0).sum(-1) / (waveform_truth - waveform_estim).pow(2.0).sum(-1) )
        if self.reduction == 'sum':
            return torch.sum(error)
        elif self.reduction == 'mean':
            return torch.mean(error)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")        


################################################################################################################################
class LossSTFT(Loss):
    def __init__(
            self,
            reduction: str = 'mean',
            win_size: int = 1024,
            hop_size: int = 512,
            win_func: str = 'hann',
            comp: float = 1.0,
            norm: int | float = 2,
            beta: float = 0.25,
            device: str | torch.device = 'cpu',
            **kwargs,
    ):
        """Criterion that measures the error between estimated and ground-truth spectrograms.

        Args:
            reduction (str, optional): Specifies the reduction to apply to the output.
                Options: `'mean'`, `'sum'`. Default: `'mean'`.
            win_size (int, optional): Window size for the STFT. Default: `1024`.
            hop_size (int, optional): Hop size for the STFT. Default: `512`.
            win_func (str, optional): Window function for the STFT. Default: `'hann'`.
            comp (float): Compression factor for the STFT error. Default: `1.0`.
            norm (int, float): Norm for the STFT error. Default: `2`.
            beta (float): Weight for the complex part of the STFT error. Default: `0.25`.
            device (str): PyTorch device to use. Default: `'cpu'`.
            **kwargs: Additional keyword arguments to pass to the STFT.
        """
        super().__init__()
        self.reduction = reduction
        self.comp = comp
        self.norm = norm
        self.beta = beta
        self.stft = fourier.STFT(
            win_size = win_size,
            hop_size = hop_size,
            win_func = win_func,
            complex_output = bool(beta),
            magnitude_only = not bool(beta),
            device = device,
            **kwargs,
        )

    def forward(
            self,
            waveform_estim: torch.Tensor,
            waveform_truth: torch.Tensor,
    ):
        """
        Args:
            waveform_truth (torch.Tensor): Input waveform tensor with size `[B, L]`.
            waveform_estim (torch.Tensor): Target waveform tensor with size `[B, L]`.
        Returns:
            torch.Tensor: Output loss tensor with size `[]`.
        """
        specgram_truth = self.stft(waveform_truth)
        specgram_estim = self.stft(waveform_estim)

        if self.beta:
            # Complex spectrograms.
            specgram_truth = torch.pow(specgram_truth.abs() + 1e-8, self.comp) * torch.exp(1j * specgram_truth.angle())
            specgram_estim = torch.pow(specgram_estim.abs() + 1e-8, self.comp) * torch.exp(1j * specgram_estim.angle())
            complex_specgram_error = torch.pow(torch.abs(specgram_estim - specgram_truth), self.norm)
            magnitude_specgram_error = torch.pow(specgram_estim.abs() - specgram_truth.abs(), self.norm)
            error = self.beta * complex_specgram_error + (1-self.beta) * magnitude_specgram_error
        else:
            # Magnitude-only spectrograms.
            specgram_truth = torch.pow(specgram_truth, self.comp)
            specgram_estim = torch.pow(specgram_estim, self.comp)
            error = torch.pow(torch.abs(specgram_estim - specgram_truth), self.norm)

        if self.reduction == 'sum':
            return torch.sum(error)
        elif self.reduction == 'mean':
            return torch.mean(error)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")


################################################################################################################################
class LossSPEC(Loss):
    def __init__(
            self,
            stft: fourier.STFT,
            reduction: str = 'mean',
            vector_norm_ord: int | float = 2,
    ):
        """Criterion that measures the error between estimated and ground-truth spatial covariance matrices.
        
        Args:
            stft (fourier.STFT): Spectrogram transform.
            reduction (str, optional): Specifies the reduction to apply to the output.
                Options: `'mean'`, `'sum'`. Default: `'mean'`.
            vector_norm_ord (int | float, optional): Order of the vector norm.
                Default: `2`.
        """
        super().__init__()
        self.stft = stft
        self.reduction = reduction
        self.vector_norm_ord = vector_norm_ord

    def forward(
            self,
            specgram_estim: torch.Tensor,
            waveform_target: torch.Tensor,
            *args,
    ):
        """
        Args:
            specgram_estim (torch.Tensor): Input spectrogram with size `[B, F, T, M]`.
            waveform_target (torch.Tensor): Target waveform tensor with size `[B, M, L]`.
        Returns:
            torch.Tensor: Output loss tensor with size `[]`.
        """        
        specgram_target = self.stft(waveform_target).permute(0, 2, 3, 1)
        error = torch.linalg.vector_norm(specgram_estim - specgram_target, ord=self.vector_norm_ord, dim=-1)
        if self.reduction == 'sum':
            return torch.sum(error)
        elif self.reduction == 'mean':
            return torch.mean(error)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")


################################################################################################################################
class LossSCM(Loss):
    def __init__(
            self,
            stft: fourier.STFT,
            reduction: str = 'mean',
            matrix_norm_ord: str = 'fro',
            average_target_across_time: bool = False,
    ):
        """Criterion that measures the error between estimated and ground-truth spatial covariance matrices.
        
        Args:
            stft (fourier.STFT): Spectrogram transform.
            reduction (str, optional): Specifies the reduction to apply to the output.
                Options: `'mean'`, `'sum'`. Default: `'mean'`.
            matrix_norm_ord (str, optional): Order of the matrix norm.
                Default: `'fro'`.
            average_target_across_time (bool, optional): Whether to average the target across time frames.
                Default: `False`.
        """
        super().__init__()
        self.stft = stft
        self.reduction = reduction
        self.matrix_norm_ord = matrix_norm_ord
        self.average_target_across_time = average_target_across_time

    def forward(
            self,
            scm_estim: torch.Tensor,
            waveform_target: torch.Tensor,
            *args,
    ):
        """
        Args:
            scm_estim (torch.Tensor): Input SCM with size `[B, F, T, M, M]`.
            waveform_target (torch.Tensor): Target waveform tensor with size `[B, M, L]`.
        Returns:
            torch.Tensor: Output loss tensor with size `[]`.
        """
        specgram_target = self.stft(waveform_target)
        scm_target = compute_spatial_covariance_matrix(specgram_target, average_across_time=self.average_target_across_time)
        error = torch.linalg.matrix_norm(scm_estim - scm_target, ord=self.matrix_norm_ord, dim=(-2, -1))
        if self.reduction == 'sum':
            return torch.sum(error)
        elif self.reduction == 'mean':
            return torch.mean(error)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")


################################################################################################################################
class LossISCM(Loss):
    def __init__(
            self,
            stft: fourier.STFT,
            reduction: str = 'mean',
            matrix_norm_ord: str = 'fro',
            average_target_across_time: bool = False,
    ):
        """Criterion that measures the error between estimated and ground-truth inverse spatial covariance matrices.
        
        Args:
            stft (fourier.STFT): Spectrogram transform.
            reduction (str, optional): Specifies the reduction to apply to the output.
                Options: `'mean'`, `'sum'`. Default: `'mean'`.
            matrix_norm_ord (str, optional): Order of the matrix norm.
                Default: `'fro'`.
            average_target_across_time (bool, optional): Whether to average the target across time frames.
                Default: `False`.
        """
        super().__init__()
        self.stft = stft
        self.reduction = reduction
        self.matrix_norm_ord = matrix_norm_ord
        self.average_target_across_time = average_target_across_time

    def forward(
            self,
            scm_estim: torch.Tensor,
            waveform_target: torch.Tensor,
            *args,
    ):
        """
        Args:
            scm_estim (torch.Tensor): Input inverse SCM with size `[B, F, T, M, M]`.
            waveform_target (torch.Tensor): Target waveform tensor with size `[B, M, L]`.
        Returns:
            torch.Tensor: Output loss tensor with size `[]`.
        """
        specgram_target = self.stft(waveform_target)
        iscm_target = inverse_spatial_covariance_matrix(specgram_target, average_across_time=self.average_target_across_time)
        error = torch.linalg.matrix_norm(scm_estim - iscm_target, ord=self.matrix_norm_ord, dim=(-2, -1))
        if self.reduction == 'sum':
            return torch.sum(error)
        elif self.reduction == 'mean':
            return torch.mean(error)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")


################################################################################################################################
class LossGEVD(Loss):
    def __init__(
            self,
            stft: fourier.STFT,
            reduction: str = 'mean',
            vector_norm_ord: int | float = 2,
            average_target_across_time: bool = False,
    ):
        """Criterion that measures the error between estimated and ground-truth generalized eigenvectors.
        
        Args:
            stft (fourier.STFT): Spectrogram transform.
            reduction (str, optional): Specifies the reduction to apply to the output.
                Options: `'mean'`, `'sum'`. Default: `'mean'`.
            vector_norm_ord (int | float, optional): Order of the vector norm.
                Default: `2`.
            average_target_across_time (bool, optional): Whether to average the target across time frames.
                Default: `False`.
        """
        super().__init__()
        self.stft = stft
        self.reduction = reduction
        self.vector_norm_ord = vector_norm_ord
        self.average_target_across_time = average_target_across_time

    def forward(
            self,
            geigenvec_estim: torch.Tensor,
            waveform_noise: torch.Tensor,
            waveform_clean: torch.Tensor,
    ):
        """
        Args:
            geigenvec_estim (torch.Tensor): Input generalized eigenvector with size `[B, F, T, M, M]`.
            waveform_noise (torch.Tensor): Noise waveform tensor with size `[B, M, L]`.
            waveform_clean (torch.Tensor): Clean speech waveform tensor with size `[B, M, L]`.
        Returns:
            torch.Tensor: Output loss tensor with size `[]`.
        """
        specgram_clean = self.stft(waveform_clean)
        specgram_noise = self.stft(waveform_noise)
        geigenvec_target = principal_generalized_eigenvector(specgram_clean, specgram_noise, average_across_time=self.average_target_across_time)
        if self.average_target_across_time:
            error = torch.linalg.vector_norm(geigenvec_estim - geigenvec_target, ord=self.vector_norm_ord, dim=-2).sum(-1)
        else:
            error = torch.linalg.vector_norm(geigenvec_estim[..., -1] - geigenvec_target, ord=self.vector_norm_ord, dim=-1)
        if self.reduction == 'sum':
            return torch.sum(error)
        elif self.reduction == 'mean':
            return torch.mean(error)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Expected 'mean' or 'sum'.")


################################################################################################################################
def compute_spatial_covariance_matrix(
        u: torch.Tensor,
        average_across_time: bool = False,
    ):
    """Compute spatial covariance matrix of complex vector `u`."""
    mat = torch.einsum('bmft, bnft -> bftmn', u, u.conj())
    if average_across_time:        
        mat = torch.mean(mat, dim=2, keepdim=True)
    return mat


def inverse_spatial_covariance_matrix(
        u: torch.Tensor,
        rho: float = 1.0,
        eps: float = 1e-8,
        average_across_time: bool = False,
    ):
    """Compute inverse spatial covariance matrix of complex vector `u`."""
    mat = compute_spatial_covariance_matrix(u, average_across_time=average_across_time)
    if average_across_time:
        inv = torch.linalg.inv(mat)
    else:
        m = u.size(1)
        trc = torch.sum(u.permute(0,2,3,1).abs().pow(2), dim=-1, keepdim=True).unsqueeze(-1)
        eye = torch.eye(m, dtype=mat.dtype, device=mat.device).expand_as(mat)
        reg = eps + (rho/m) * trc
        eta = 1/(reg + trc)
        inv = (1/reg) * (eye - eta*mat)
    return inv


def principal_generalized_eigenvector(
        u_a: torch.Tensor, 
        u_b: torch.Tensor, 
        rho: float = 1.0, 
        eps: float = 1e-8,
        average_across_time: bool = False,
    ):
    """Compute principal generalized eigenvector(s) of SCMs of complex vectors `u_a` and `u_b`."""
    if average_across_time:
        mat_a = compute_spatial_covariance_matrix(u_a, average_across_time=True)
        mat_b = compute_spatial_covariance_matrix(u_b, average_across_time=True)
        l, _ = torch.linalg.cholesky_ex(mat_b)
        c = torch.linalg.solve(l, mat_a)
        c = torch.linalg.solve(l.mH, c, left=False)
        _, vec = torch.linalg.eigh(c)
        vec = torch.linalg.solve(l.mH, vec)
        vec = vec / (eps + torch.linalg.vector_norm(vec, dim=-2, keepdim=True))
    else:
        inv = inverse_spatial_covariance_matrix(u_b, rho=rho, average_across_time=False)
        col = torch.einsum('bmft, bft -> bftm', u_a, u_a[:,0].conj())
        vec = torch.einsum('bftmn, bftn -> bftm', inv, col)
        vec = vec / (eps + torch.linalg.vector_norm(vec, dim=-1, keepdim=True))
    return vec