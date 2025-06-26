import torch
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality as PESQ
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility as STOI
from torchmetrics.audio.sdr import SignalDistortionRatio as SDR
from torchmetrics.audio.sdr import ScaleInvariantSignalDistortionRatio as SISDR
from torchmetrics.audio.snr import SignalNoiseRatio as SNR
from torchmetrics.audio.snr import ScaleInvariantSignalNoiseRatio as SISNR


def get_metric(metric_name, sr=16000):
    """Returns the desired metric object based on the metric name and sampling rate."""
    metric_mapping = {
        'PESQ': PESQ(fs=sr, mode='wb'),
        'STOI': STOI(fs=sr, extended=False),
        'ESTOI': STOI(fs=sr, extended=True),
        'SDR': SDR(),
        'SISDR': SISDR(),
        'SNR': SNR(),
        'SISNR': SISNR(),        
    }
    if metric_name not in metric_mapping:
        raise ValueError(f"Unknown metric: '{metric_name}'.")
    return metric_mapping[metric_name.upper()]


def compute_one_metric(
        metric_name: str, 
        reference: torch.Tensor, 
        estimate: torch.Tensor, 
        device: str | torch.device = 'cpu',
    ):
    """Compute a single metric given reference and estimate signals.
    
    Args:
        metric_name (str): Metric name.
            Example: `['SDR', 'STOI']`.
        reference (torch.Tensor): Reference signal.
            Tensor of size `[batch_size, waveform_length]`.    
        estimate (torch.Tensor): Estimate signals.
            Tensor of size `[batch_size, waveform_length]`. 
        device (str | torch.device, optional): Device to compute the metrics.
            Default: `'cpu'`.

    Returns:
        torch.Tensor: Metric score.
            Tensor of size `[batch_size]`.
    """
    metric = get_metric(metric_name).to(device=device)
    estimate = torch.nan_to_num(estimate, nan=0.0)
    return metric(target=reference, preds=estimate)


def compute_all_metrics(
        metric_name_list: list[str], 
        reference: torch.Tensor, 
        estimate: torch.Tensor, 
        device: str | torch.device = 'cpu',
    ):
    """Compute multiple metrics given reference and estimate signals.

    Args:
        metric_name_list (list[str]): List of metric names.
            Example: `['SDR', 'STOI']`.
        reference (torch.Tensor): Reference signal.
            Tensor of size `[..., L]`.    
        estimate (torch.Tensor): Estimate signals.
            Tensor of size `[..., L]`.
        device (str | torch.device, optional): Device to compute the metrics.
            Default: `'cpu'`.

    Returns:
        torch.Tensor: Tensor of size `(num_metrics, batch_size)`.
    """
    batch_size, num_metrics = len(reference), len(metric_name_list)
    metric_score = torch.empty(size=(num_metrics, batch_size), device=device)
    for i, metric_name in enumerate(metric_name_list):
        for b in range(batch_size):
            metric_score[i,b] = compute_one_metric(metric_name, reference[b], estimate[b], device=device)
    return metric_score