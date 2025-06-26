import torch
import torchaudio
import os
import pathlib

class MixturesDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            root: str | pathlib.Path, 
            subset: str | pathlib.Path,
            num_samples: int | None = None,
            fixed_ref_mic: int | None = 0,
            load_all: bool = False,
            file_ext: str = 'flac',
    ):
        """Dataset class to handle the MIXTURES database.

        Args:
            root (str, pathlib.Path): Path to the root directory of the dataset.
            subset (str, pathlib.Path): Name of the subset to load.
            num_samples (int, optional): Number of samples to load. Default: `None` (all samples). 
            fixed_ref_mic (int, optional): Index for the reference microphone. If `None`, chosen at random. Default: `0`.
            load_all (bool, optional): Whether to load all signals. If `False`, only mixture and ground-truth signals are loaded. Default: `False`.
            file_ext (str, optional): Extension of the audio files. Default: `'flac'`.
        """
        super().__init__()
        self.subset_name = subset
        self.subset_path = os.path.join(root, subset)
        self.num_samples = num_samples
        self.fixed_ref_mic = fixed_ref_mic
        self.load_all = load_all
        self.file_ext = file_ext        
        # Collect sample names.
        self.sample_names = self._collect_samples()
    
    def _collect_samples(self):
        sample_names = [s for s in os.listdir(self.subset_path) if not s.startswith('.')]
        return sorted(sample_names)[:self.num_samples]

    def _get_file_path(self, sample_name, file_name):
        return os.path.join(self.subset_path, sample_name, f'{file_name}.{self.file_ext}')

    def _load_sample(self, n: int):
        sample_name = self.sample_names[n]

        # Load mixture audio signals.
        file_path_mixtr = self._get_file_path(sample_name, 'mixtr')
        waveform_mixtr, sr_mixtr = torchaudio.load(file_path_mixtr)

        # Generate a fixed or random reference microphone vector.
        num_channels = waveform_mixtr.size(0)
        if self.fixed_ref_mic is None:
            ref_mic = int(torch.randint(0, num_channels, (1,)))
        else:
            ref_mic = self.fixed_ref_mic
        ref_mic_vect = torch.eye(num_channels)[ref_mic]

        # Load ground-truth audio signals.
        file_path_truth = self._get_file_path(sample_name, 'clean')
        waveform_truth, sr_truth = torchaudio.load(file_path_truth)
        waveform_truth = waveform_truth[ref_mic]    

        # Assert audio file consistency.
        assert (sr_mixtr == sr_truth), f"Sampling rates do not match across files."
        assert (waveform_mixtr.size(-1) == waveform_truth.size(-1)), "Signal length does not match across files."
        
        # Load additional audio signals and return.
        if self.load_all:
            file_path_clean = self._get_file_path(sample_name, 'clean')
            waveform_clean, _ = torchaudio.load(file_path_clean)
            file_path_noise = self._get_file_path(sample_name, 'noise')
            waveform_noise, _ = torchaudio.load(file_path_noise)
            return waveform_mixtr, waveform_truth.squeeze(0), waveform_clean, waveform_noise, ref_mic_vect, sample_name
        else:
            return waveform_mixtr, waveform_truth.squeeze(0), ref_mic_vect

    def __getitem__(self, n: int):
        """ Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded.

        Returns:
            torch.Tensor: Noisy mixture signal. Size `[M, L]`.
            torch.Tensor: Ground-truth speech signal. Size `[L]`.
            torch.Tensor, optional: Clean speech signal. Size `[M, L]`.
            torch.Tensor, optional: Noise signal. Size `[M, L]`.
            torch.Tensor: Reference microphone vector. Size `[M]`.
            str, optional: Sample name.
        """
        return self._load_sample(n)  

    def __len__(self):
        return len(self.sample_names)
    

def dataloaders(
        root: str | pathlib.Path, 
        batch_size: int, 
        num_samples_trn: int | None = None, 
        num_samples_val: int | None = None, 
        num_samples_tst: int | None = None, 
        **dataset_kwargs,
    ):
    """Define PyTorch DataLoaders for the training and validation subsets.

    Args:
        root (str, pathlib.Path): Path to the root directory of the database.
        batch_size (int): Batch size for PyTorch DataLoaders.
        num_samples_trn (int, optional): Number of samples to load for the training dataset. Default: `None` (all samples).
        num_samples_val (int, optional): Number of samples to load for the validation dataset. Default: `None` (all samples).
        num_samples_tst (int, optional): Number of samples to load for the test dataset. Default: `None` (all samples).
        **dataset_kwargs: Other keyword arguments for the Datasets.

    Returns:
        loaders (dict): Dictionary containing the DataLoaders.
    """
    trn_dataset = MixturesDataset(root, 'trn', num_samples=num_samples_trn, **dataset_kwargs)
    val_dataset = MixturesDataset(root, 'val', num_samples=num_samples_val, **dataset_kwargs)
    tst_dataset = MixturesDataset(root, 'tst', num_samples=num_samples_tst, **dataset_kwargs)
    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=batch_size, shuffle=False)
    return {'trn_loader': trn_loader, 'val_loader': val_loader, 'tst_loader': tst_loader}
