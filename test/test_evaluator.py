
from realign.evaluators import evaluator


if __name__ == '__main__':

    @evaluator
    def simple_add(a, b):
        return a + b

    print('score', simple_add(1, 2))
    print(simple_add.prev_run)