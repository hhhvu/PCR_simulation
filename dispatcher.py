# Added my dispatcher from before 

import argparse
import json
import random
import os, subprocess
from csv import DictWriter
import multiprocessing
from vectorizer import Vectorizer
from logistic_regression import LogisticRegression
from main import load_data
from sklearn.metrics import roc_auc_score

import numpy as np

#Run scripts
# python dispatcher.py --config_path grid_search.json --num_workers 5 --log_dir logs_v2 --grid_search_results_path grid_results_q1_2_v2.csv
# python dispatcher.py --config_path grid_search_abla.json --num_workers 50 --log_dir logs_abla --grid_search_results_path grid_results_q1_2_abla.csv
# python dispatcher.py --config_path grid_search_abla.json --num_workers 75 --log_dir logs_abla_mulcov --grid_search_results_path grid_results_q1_3_abla_mulcov_v2.csv

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config_path",
        type=str,
        default="grid_search.json",
        help="Location of config file"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of processes to run in parallel"
    )

    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Location of experiment logs and results"
    )

    parser.add_argument(
        "--grid_search_results_path",
        default="grid_results_q1_2.csv",
        help="Where to save grid search results"
    )

    return parser

def get_experiment_list(config: dict) -> (list(dict)):
    '''
    Parses an experiment config, and creates jobs. For flags that are expected to be a single item, but the config contains a list, this will return one job for each item in the list.
    :config - experiment_config
    {'learning_rate': [0.0001], 'batch_size': [256], 'num_epochs': [10], 'regularization_lambda': [0]}

    returns: jobs - a list of dicts, each of which encapsulates one job.
        *Example: {learning_rate: 0.001 , batch_size: 16 ...}
    '''

    jobs = [{}]

    for key, values in config.items():
        # If the value is a single item list, just add it to each job
        if len(values) == 1:
            for job in jobs:
                job[key] = values[0]
        else:
            # Otherwise, make new jobs for each value
            new_jobs = []
            for value in values:
                for existing_job in jobs:
                    new_job = existing_job.copy() # Make a copy of the existing job
                    new_job[key] = value          # Set the new parameter value
                    new_jobs.append(new_job)      # Add the new job to our list of new jobs
            jobs = new_jobs                       # Replace the existing jobs with the new ones

    return jobs


def worker(args: argparse.Namespace, job_queue: multiprocessing.Queue, done_queue: multiprocessing.Queue):
    '''
    Worker thread for each worker. Consumes all jobs and pushes results to done_queue.
    :args - command line args
    :job_queue - queue of available jobs.
    :done_queue - queue where to push results.
    '''
    while not job_queue.empty():
        params = job_queue.get()
        if params is None:
            return
        done_queue.put(
            launch_experiment(args, params))

def launch_experiment(args: argparse.Namespace, experiment_config: dict) ->  dict:
    '''
    Launch an experiment and direct logs and results to a unique filepath.
    :configs: flags to use for this model run. Will be fed into
    scripts/main.py

    returns: flags for this experiment as well as result metrics
    '''

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # The command to run the script
    lr, lamb, batch, epo = experiment_config['learning_rate'], experiment_config['regularization_lambda'], experiment_config['batch_size'], experiment_config['num_epochs']
    results_path = f'{args.log_dir}/results_{lr}_{lamb}_{batch}_{epo}_ablation.json'
    command = ['python', 'main.py', '--learning_rate', str(lr), '--regularization_lambda', str(lamb), '--batch_size', str(batch), '--num_epochs', str(epo), '--results_path', results_path]

    # Run the command and capture the output
    try:
        results = subprocess.run(command, stdout=subprocess.PIPE)

        # Print the output
        #print(results.stdout.decode('utf-8'))

        # TODO: Parse the results from the experiment and return them as a dict
        with open(results_path, 'r') as file:
            results = json.load(file)
        
        experiment_config['coefficient'] = results['coefficient'][0]
        experiment_config['train_auc'] = results['train_auc']
        experiment_config['val_auc'] = results['val_auc']
        experiment_config['train_loss'] = results['train_losses'][-1]
        experiment_config['val_loss'] = results['val_losses'][-1]

        results = experiment_config
        print('########## Experiment results ##########')
        print(results['regularization_lambda'])
        print(results['num_epochs'])
        print(results['coefficient'])
        print(results['train_loss'])
        print(results['train_auc'])
        print(results['val_auc'])
    except Exception as e:
        experiment_config['coefficient'] =  float("NaN")
        experiment_config['train_auc'] =  float("NaN")
        experiment_config['val_auc'] =  float("NaN")
        experiment_config['train_loss'] =  float("NaN")
        experiment_config['val_loss'] =  float("NaN")

        results = experiment_config
        print('########## Experiment results ##########')
        print(results['regularization_lambda'])
        print(results['num_epochs'])
        print(results['coefficient'])
        print(results['train_loss'])
        print(results['train_auc'])
        print(results['val_auc'])

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def main(args: argparse.Namespace) -> dict:
    print(args)
    config = json.load(open(args.config_path, "r"))
    print("Starting grid search with the following config:")
    print(config)

    # TODO: From config, generate a list of experiments to run
    experiments = get_experiment_list(config)
    random.shuffle(experiments)

    job_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()

    for exper in experiments:
        job_queue.put(exper)

    print("Launching dispatcher with {} experiments and {} workers".format(len(experiments), args.num_workers))

    # TODO: Define worker fn to launch an experiment as a separate process.
    for _ in range(args.num_workers):
        multiprocessing.Process(target=worker, args=(args, job_queue, done_queue)).start()

    # Accumualte results into a list of dicts
    grid_search_results = []
    for _ in range(len(experiments)):
        grid_search_results.append(done_queue.get())

    keys = grid_search_results[0].keys()

    print("Saving results to {}".format(args.grid_search_results_path))

    writer = DictWriter(open(args.grid_search_results_path, 'w'), keys)
    writer.writeheader()
    writer.writerows(grid_search_results)

    print("Done")

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
