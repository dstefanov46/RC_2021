import argparse
from datetime import datetime
from pathlib import Path
import pickle

from gluonts.model.deephier import DeepHierEstimator

import experiment
import utils

epoch_set = [10, 25, 50]
context_len_dict = {'tourismsmall': [2, 3, 4],
                    'tourismlarge': [2, 3, 4],
                    'labour': [2, 3, 4],
                    'traffic': [15, 25, 40, 60],
                    'wiki2': [15, 25, 40, 60]}
prediction_len_dict = {'tourismsmall': 8,
                       'tourismlarge': 12,
                       'labour': 8,
                       'traffic': 1,
                       'wiki2': 1}
warm_start_dict = {'DeepVAR': 0,
                   'DeepVARPlus': 0,
                   'HierE2E': 0.1}

if __name__ == "__main__":
    """
    Test different hyperparameter configurations for the 3 neural models (DeepVAR, DeepVARPlus, HierE2E):
    
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_runs", type=int, required=True)

    args, _ = parser.parse_known_args()

    method = args.method
    dataset = args.dataset
    num_runs = args.num_runs

    warm_start_frac = warm_start_dict[method]
    if method == 'HierE2E':
        coherent_train_samples = True
    else:
        coherent_train_samples = False
    if method == 'DeepVAR':
        coherent_pred_samples = False
    else:
        coherent_pred_samples = True

    pick_incomplete = False
    if dataset == "wiki":
        dataset = "wiki2"
    elif dataset == "tourism":
        dataset = "tourismsmall"
        pick_incomplete = True
    context_lens = context_len_dict[dataset]

    if dataset == 'tourismlarge' and method == 'HierE2E':
        batch_size = 4
    else:
        batch_size = 32

    estimator = DeepHierEstimator

    for n_epochs in epoch_set:
        for context_len in context_lens:
            hyper_params = {'epochs': n_epochs,
                            'num_batches_per_epoch': 50,
                            'scaling': True,
                            'pick_incomplete': pick_incomplete,
                            'batch_size': batch_size,
                            'num_parallel_samples': 200,
                            'hybridize': False,
                            'learning_rate': 0.001,
                            'context_length': context_len * prediction_len_dict[dataset],
                            'rank': 0,
                            'assert_reconciliation': False,
                            'num_deep_models': 1,
                            'num_layers': 2,
                            'num_cells': 40,
                            'coherent_train_samples': coherent_train_samples,
                            'coherent_pred_samples': coherent_pred_samples,
                            'likelihood_weight': 1.0,
                            'CRPS_weight': 0.0,
                            'warmstart_epoch_frac': warm_start_frac}

            if method == 'HierE2E':
                extra_params = {'num_samples_for_loss': 50,
                                'sample_LH': True}
                hyper_params.update(extra_params)

            print(f"Running {method} on {dataset} dataset.")
            print(hyper_params)

            job_config=dict(
                metrics=["mean_wQuantileLoss"],
                validation=True,
            )

            results_path = f"./experiments/results/{method}/{dataset}/epochs_{n_epochs}_context_len_factor_{context_len}"
            Path(results_path).mkdir(parents=True, exist_ok=True)

            agg_metrics_ls = []
            level_wise_agg_metrics_ls = []
            for i in range(num_runs):
                print(f"********* Run {i+1} *********")
                agg_metrics, level_wise_agg_metrics = experiment.main(
                    method=method,
                    dataset_path=f'./experiments/data/{dataset}',
                    estimator=estimator,
                    hyper_params=hyper_params,
                    job_config=job_config
                )

                agg_metrics_ls.append(agg_metrics)
                level_wise_agg_metrics_ls.append(level_wise_agg_metrics)

                # Save results
                unique_id = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
                with open(f"{results_path}/run_{unique_id}.pkl", "wb") as fp:
                    pickle.dump([agg_metrics, level_wise_agg_metrics], fp)

            utils.print_results(agg_metrics_ls=agg_metrics_ls, level_wise_agg_metrics_ls=level_wise_agg_metrics_ls)
