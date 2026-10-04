class AddConstant:
    def __init__ (self, c):
        self.c = c
    def __call__(self, x):
        return x + self.c
add_five = AddConstant(5)
print(add_five(10))
