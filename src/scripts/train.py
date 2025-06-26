import torch
import os
import sys
import pathlib
import yaml
import thop
import datetime
sys.path.append(os.path.abspath(''))
import src.utils.losses
import src.utils.solver
import src.utils.dataset
import FaSNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Load configuration file.
CONFIG_PATH = pathlib.Path('src', 'config', 'training.yaml')
with CONFIG_PATH.open('r') as f:
    CONFIG = yaml.safe_load(f)
SOLVER_PATH = pathlib.Path('out', 'pkls', f"{datetime.datetime.today().strftime('%m%d')}_{CONFIG['model']['model_name']}.pkl")
SRC_LOG_PATH = pathlib.Path('.', 'training.log')
DST_LOG_PATH = pathlib.Path('out', 'logs', f"{datetime.datetime.today().strftime('%m%d')}_{CONFIG['model']['model_name']}.log")


def train(config: dict, database_root: str | pathlib.Path):
    """Define model, loaders, optimizer, criterion, scheduler, solver and launch training.

    Args:
        config (dict): Dictionary containing the training parameters.
        database_root (str, pathlib.Path): Path to the root of the training database.
    Returns:
        solver (Solver): Solver instance containing the training information, e.g. model and loss history.
    """
    # Define model.
    model = FaSNet.FaSNet_TAC(
        **config['model']['model_args'],
    )
    # Define criterion.
    criterion = src.utils.losses.MultiLoss(

        waveform_criterion = getattr(src.utils.losses, config['criterion']['waveform_criterion_name'])(
            **config['criterion']['waveform_criterion_args'],
        ) if config['criterion']['waveform_criterion_name'] else None,
        waveform_criterion_scale = config['criterion']['waveform_criterion_scale'],

        specgram_criterion = getattr(src.utils.losses, config['criterion']['specgram_criterion_name'])(
            **config['criterion']['specgram_criterion_args'],
            device = DEVICE,
        ) if config['criterion']['specgram_criterion_name'] else None,
        specgram_criterion_scale = config['criterion']['specgram_criterion_scale'],

    )
    # Define optimizer.
    optimizer = getattr(torch.optim, config['optimizer']['optimizer_name'])(
        params = model.parameters(),
        **config['optimizer']['optimizer_args'],
    )
    # Define scheduler.
    scheduler = getattr(torch.optim.lr_scheduler, config['scheduler']['scheduler_name'])(
        optimizer = optimizer,
        **config['scheduler']['scheduler_args'],
    )
    # Define dataloaders.
    dataloaders = src.utils.dataset.dataloaders(
        root = database_root, 
        batch_size = config['training']['batch_size'],
        num_samples_trn = config['dataloader']['num_samples_trn'], 
        num_samples_val = config['dataloader']['num_samples_val'], 
        num_samples_tst = 0,
        fixed_ref_mic = config['dataloader']['fixed_ref_mic'],
        load_all = False,
    )
    # Print model specifications.
    print('---------------------------------------')
    input_mixtr = torch.randn(1, 4, 64000)
    input_ref_mic = torch.tensor([[0]], dtype=torch.int64)
    num_macs, num_params, *_ = thop.profile(model, inputs=(input_mixtr, input_ref_mic), verbose=False)
    print(f"Num. Parameters: {int(num_params):,} | Num. MACs: {int(num_macs):,}.")
    # Define solver.
    model.to(DEVICE)
    solver = src.utils.solver.Solver(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        dataloaders, 
        config = config,
        path = SOLVER_PATH,
        device = DEVICE,
    )
    os.makedirs(os.path.dirname(SOLVER_PATH), exist_ok=True)
    
    # Train model.
    solver = solver.train()

    return solver


def prepare_log_file(log_path_src: str | pathlib.Path, log_path_dst: str | pathlib.Path):
    # Handle exceptions.
    def excepthook(exc_type, exc_value, exc_traceback):
        if isinstance(exc_value, KeyboardInterrupt):
            print("\n*** TRAINING INTERRUPTED BY USER ***")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            os.replace(src=log_path_src, dst=log_path_dst)
        else:
            print("\n*** TRAINING INTERRUPTED BY EXCEPTION***")
            print(f"{exc_type.__name__}: {exc_value}")
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            os.replace(src=log_path_src, dst=log_path_dst)
    # Redirect stdout to a log file.
    # os.makedirs(os.path.dirname(log_path_src), exist_ok=True)
    sys.stdout = open(log_path_src, 'wt')            
    sys.excepthook = excepthook


if __name__ == '__main__':

    # set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Redirect stdout to a default log file. 
    prepare_log_file(SRC_LOG_PATH, DST_LOG_PATH)
    print("*** START TRAINING ***\n")

    root = "/home/ovistetom/Documents/Databases_Local/MIXTURES/standard"
    # root = os.path.join('database', 'MIXTURES')
    # Train the model.
    solver = train(config=CONFIG, database_root=root)
    
    # Close and save log file.
    print("\n*** FINISHED TRAINING ***")
    sys.stdout.close()
    os.replace(src=SRC_LOG_PATH, dst=DST_LOG_PATH)    