from itertools import chain

from realign import evaluator

@evaluator
def flatten(list_of_lists):
    return list(chain(*list_of_lists))