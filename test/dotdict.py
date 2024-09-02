from realign.utils import dotdict

# Usage example
d = dotdict({'a': 1, 'b': {'c': 2}})
print(d.a)  # Output: 1
print(d.b.c)  # Output: 2
d.new_key = 'value'
print(d['new_key'])  # Output: value