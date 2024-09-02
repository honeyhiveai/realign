
from realign import evaluator, aevaluator, config, run_async

config.yaml = '''
evaluators:
    simple_sub:
        asserts: off
        checker: numrange(value, target)
        target: [0, 3]
'''

    
@evaluator(c=10)
def simple_add(a, b, c=0):
    return a + b + c

@evaluator
def simple_sub(a, b):
    return a - b

# assert simple_add(1, 2) == 3



evaluator['simple_add'](1, 2)


evaluator['simple_sub'].raw(1, 4)


print(simple_add.settings)

print(evaluator['simple_sub'].settings)
print(simple_add.kwargs)

print(simple_add(10, 20, checker=''))


@evaluator
def the_sum(x):
    return sum(x)

print(run_async(
    simple_add(10, 20, checker='', repeat=2, aggregate='the_sum')
))
