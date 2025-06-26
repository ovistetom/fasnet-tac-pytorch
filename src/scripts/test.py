import torch
import pandas as pd
import tqdm
import time
import os
import sys
# Add parent directory to the path.
sys.path.append(os.path.abspath(''))
import src.utils.dataset
import src.utils.metrics
import src.utils.unpickler
import FaSNet


# Define evaluation parameters.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = '0623_fasnet_tac'
MODEL_PATH = os.path.join('out', 'pkls', f'{MODEL_NAME}.pkl')
METRICS = ['PESQ', 'STOI', 'SDR', 'ESTOI']


def test(model_path, database_root):

    # Define dataloaders.
    dataloaders = src.utils.dataset.dataloaders(
        root=database_root,
        batch_size=1,
        num_samples_trn = 0, 
        num_samples_val = 0, 
        num_samples_tst = 6000, 
        fixed_ref_mic = 0,
        load_all = True,
    )

    # Load model.
    with open(model_path, 'rb') as solver_file:
        solver_package = src.utils.unpickler.UnpicklerCPU(solver_file).load()
    keys = list(solver_package['model_state_dict'].keys())
    for key in keys:
        if 'total_params' in key or 'total_ops' in key:
            solver_package['model_state_dict'].pop(key);

    model = FaSNet.FaSNet_TAC(**solver_package['model_args']).to(DEVICE)
    model.load_state_dict(solver_package['model_state_dict'])

    # Evaluate model.
    model.eval()
    data = {}    
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(dataloaders['tst_loader'], "Testing epoch")):

            # Get the inputs and targets.
            batch_mixtr, batch_truth, _, _, _, sample_name = batch
            batch_mixtr = batch_mixtr.to(DEVICE)
            batch_truth = batch_truth.to(DEVICE)
            batch_ref_mic = torch.zeros(1, dtype=torch.int).to(DEVICE)

            # Forward pass.
            tic = time.time()
            batch_estim, *_ = model(batch_mixtr, batch_ref_mic)
            toc = time.time()

            # Compute metrics.
            processing_time = toc-tic
            metric_score_list = src.utils.metrics.compute_all_metrics(METRICS, reference=batch_truth, estimate=batch_estim, device=DEVICE)
            
            # Fetch metadata about the sample.
            sample_name = sample_name[0]
            sample_path = os.path.join(database_root, 'tst', sample_name, 'metadata.txt')
            with open(sample_path, 'r') as metadata_file:
                metadata = metadata_file.readlines()            
            sample_data = [l.split(',')[1].strip() for l in metadata]
            sample_data = [sample_name, *sample_data[0:3], float(sample_data[3]), float(sample_data[4]), sample_data[5].startswith('T')]

            # Define Pandas row.
            row = sample_data + metric_score_list.flatten().tolist() + [processing_time]
            data[i] = row

    return data


if __name__ == '__main__':
    # Empty the GPU cache.
    torch.cuda.empty_cache()
    torch.autograd.set_detect_anomaly(True)

    root = "/home/ovistetom/Documents/Databases_Local/MIXTURES/standard"
    # root = os.path.join('database', 'MIXTURES')
    # Evaluate model.
    data = test(MODEL_PATH, root)

    # Convert to Pandas DataFrame.
    df = pd.DataFrame.from_dict(
        data, 
        orient = 'index', 
        columns = ['SAMPLE_NAME', 'SPEAK', 'DISTR', 'NOISE', 'DISTR_SNR', 'NOISE_SNR', 'ECHO', *METRICS, 'PROCESSING_TIME'],
    )

    # Save DataFrame to CSV file.
    os.makedirs(os.path.join('out', 'csvs'), exist_ok=True)
    csv_file_path = os.path.join('out', 'csvs', f'{MODEL_NAME}.csv')
    df.to_csv(csv_file_path)
    print(f"Data saved to '{csv_file_path}'.")
