def print_kwargs(**kwargs):
    print(kwargs)

print_kwargs(a=1, b="hello", c=[1, 2, 3])
# 출력: {'a': 1, 'b': 'hello', 'c': [1, 2, 3]}
