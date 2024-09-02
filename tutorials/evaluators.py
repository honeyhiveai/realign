from realign import evaluator, config

config.yaml = '''
evaluators:
    tweet_char_count:
        checker: numrange
        target: (0, 280]
        asserts: on
'''

@evaluator
def tweet_char_count(text: str) -> int:
    return len(text)

print(tweet_char_count('hello')) # True since score of 5 is in range (0, 280]


# assertion fails since score of 500 is out of range (0, 280]
try:
    tweet_char_count('hello'*100) 
except AssertionError as e:
    pass
else:
    raise Exception('AssertionError not raised')

# raw ignores the config and calls the decorated function directly
print(tweet_char_count.raw('hello')) # 5
print(tweet_char_count.raw('hello'*100)) # 500

# changing the config during runtime
print(tweet_char_count('hello'*100, target='(0,10000]')) # True since score of 500 is in range (0, 10000]
print(tweet_char_count('hello'*100, asserts=False)) # False, but no assertion error
print(tweet_char_count('hello'*100, checker='value == 500')) # True since score == 500
print(tweet_char_count('hello'*100, checker='value == target', target=500)) # True since score == target = 500

# you can also reference by key
print(evaluator['tweet_char_count']('hello')) # goes through tweet_char_count config

# you can set configs during runtime
tweet_char_count.config['checker'] = 'value > 10'
tweet_char_count.config['asserts'] = False
print(tweet_char_count.config)

print(tweet_char_count('hello')) # False since score of 5 is not > 10
print(tweet_char_count('hello'*100)) # True since score of 500 is > 10

tweet_char_count.config.clear()
print(tweet_char_count('hello')) # True
print(tweet_char_count('hello'*100, asserts=False)) # False since score of 500 not in (0, 280]

# you can also set configs using the decorator

@evaluator(checker='value > 100', asserts=False)
def tweet_char_count(text: str) -> int:
    return len(text)

print(tweet_char_count.config)
print(tweet_char_count('hello')) # False
print(tweet_char_count('hello'*100)) # True

# you can access your config dict using the config module
print(config['evaluators']['tweet_char_count']) # config

# you can also access default config dict using config.default
print(config.default['evaluators']['elo_ratings']) # config

# you can also call default evaluators directly
print(evaluator.llm_rating_json('hello'))