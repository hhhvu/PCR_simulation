import argparse
import time
import json
import random
import os, subprocess
import numpy as np
from csv import DictWriter
import multiprocessing


#Run scripts
# CUDA_VISIBLE_DEVICES=6,7 python scripts/pcr_dispatcher.py --config_path configs/fusion_sweep.json --num_workers 1

#python scripts/pcr_dispatcher.py --config_path configs/fusion_sweep.json --num_workers 4
#python scripts/pcr_dispatcher.py --config_path configs/gene_fusion_sweep.json --num_workers 4

def add_main_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/toy_mlp_sweep.json",
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
        default="logs/grid_results_fusion.csv",
        help="Where to save grid search results"
    )

    parser.add_argument(
        "--experiment_name",
        default="grid_search",
        help="How to name the experiments in wandb"
    )

    return parser

def get_experiment_list(config: dict):
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


# returns: flags for this experiment as well as result metrics
def launch_experiment(args: argparse.Namespace, experiment_config: dict) ->  dict:
    '''
    Launch an experiment and direct results to wandb
    :configs: flags to use for this model run. Will be fed into scripts/main.py
    '''

    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    # The command to run the script
    command = ['python', 'scripts/pcr_train_script.py', '--train', '--grid_search']

    unique_suffix = str(int(time.time()))  # Using timestamp
    experiment_name = f"{unique_suffix}"
    model_name = experiment_config['main.model_name']
    for key, value in experiment_config.items():
        key = key.split('.')[-1] if key.startswith('main.') else key
        command.extend(['--' + key, str(value)])
    command.extend(['--experiment_name', model_name + '_grid-search_' + experiment_name])

    # Run the command and capture the output
    subprocess.run(command, stdout=subprocess.PIPE)

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
        process = multiprocessing.Process(target=worker, args=(args, job_queue, done_queue)).start()
        # TO DELETE: changes for mac
        # process.start()
        # process.join()
    
    print("Done")

if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
