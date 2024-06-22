"""
가변 인자(*args)와 키워드 인자(**kwargs) 예제
"""

class MyClass:
    def __init__(self, name):
        self.name = name

    def print_info(self, *args, **kwargs):
        print(f"Name: {self.name}")
        print("Additional arguments:")
        for arg in args:
            print(f" - {arg}")
        print("Keyword arguments:")
        for key, value in kwargs.items():
            print(f" - {key}: {value}")

# 클래스 인스턴스 생성
my_object = MyClass("Alice")

# 다양한 인자와 함께 메서드 호출
my_object.print_info(25, "Engineer", location="New York", hobby="Reading")
