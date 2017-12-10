"""
Command-line argument parsing.
"""

import argparse

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrained', help='evaluate a pre-trained model',
                        action='store_true', default=False)
    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--checkpoint', help='checkpoint directory', default='model_checkpoint')
    parser.add_argument('--classes', help='number of classes per inner task', default=5, type=int)
    parser.add_argument('--shots', help='number of examples per class', default=5, type=int)
    parser.add_argument('--inner-batch', help='inner batch size', default=5, type=int)
    parser.add_argument('--inner-iters', help='inner iterations', default=20, type=int)
    parser.add_argument('--meta-step', help='meta-training step size', default=0.1, type=float)
    parser.add_argument('--meta-batch', help='meta-training batch size', default=1, type=int)
    parser.add_argument('--meta-iters', help='meta-training iterations', default=400000, type=int)
    parser.add_argument('--eval-batch', help='eval inner batch size', default=5, type=int)
    parser.add_argument('--eval-iters', help='eval inner iterations', default=50, type=int)
    parser.add_argument('--eval-samples', help='evaluation samples', default=10000, type=int)
    parser.add_argument('--eval-interval', help='train steps per eval', default=10, type=int)
    return parser

def train_kwargs(parsed_args):
    """
    Build kwargs for the train() function from the parsed
    command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'inner_batch_size': parsed_args.inner_batch,
        'inner_iters': parsed_args.inner_iters,
        'meta_step_size': parsed_args.meta_step,
        'meta_batch_size': parsed_args.meta_batch,
        'meta_iters': parsed_args.meta_iters,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'eval_interval': parsed_args.eval_interval,
    }

def evaluate_kwargs(parsed_args):
    """
    Build kwargs for the evaluate() function from the
    parsed command-line arguments.
    """
    return {
        'num_classes': parsed_args.classes,
        'num_shots': parsed_args.shots,
        'eval_inner_batch_size': parsed_args.eval_batch,
        'eval_inner_iters': parsed_args.eval_iters,
        'num_samples': parsed_args.eval_samples
    }
