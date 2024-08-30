from realign import Context

ctx = Context(3)
ctx.tweet = 'hello'
ctx['blah'] = 'world'
print(ctx.tweet)
print(ctx['blah'])
print(ctx)