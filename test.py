def example_function(**kwargs):
    for key, value in kwargs.items():
        print(f"{key} = {value}")

# kwargs에 여러 키워드 인수 추가
kwargs = {
    "name": "Alice",
    "age": 30,
    "city": "Seoul"
}
example_function(**kwargs)
print(**kwargs)