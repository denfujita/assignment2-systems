import timeit

from cs336_basics.model import BasicsTransformerLM


def benchmarking_script(num_warmups: int, ): 
    """
    A script that will initialize a basics Transformer model with the given
    hyperparameters, create a random batch of data, and time forward-only, forward-and-
    backward, and full training steps that include the optimizer step
    """
    model = BasicsTransformerLM()