# Meno: Denysenko, Andrii
# Spolupráca: 
# Použité zdroje: 
# Čas: 

# Podrobný popis je dostupný na: https://github.com/ianmagyar/introduction-to-python/blob/master/assignments/homeworks/homework07.md

# Hodnotenie: /1b


class ClassA:
    def __init__(self, value):
        self.value = value

    def foo(self):
        return self.value
    
    def bar(self):
        return "something"

# TODO: upravte definíciu tak, aby ClassB bola podtriedou ClassA
class ClassB(ClassA):
    def __init__(self, value):
        super().__init__(value)
    
    def foo(self):
        result = super().foo()
        factorial = 1
        for  i in range(1, result + 1):
            factorial *= i
        return factorial
    # TODO: prepíšte metódu foo tak, aby vrátila faktoriál
    # návratovej hodnoty implementácie z nadtriedy
    # implementácia nech obsahuje volanie metódy foo z nadtriedy


test_value = 5
testA = ClassA(test_value)
testB = ClassB(test_value)

# TODO: do komentárov napíšte, z ktorej triedy sa vykonajú implementácie metód
print(testA.foo())  # implementácia z triedy ...
print(testA.bar())  # implementácia z triedy ...
print(testB.foo())  # implementácia z triedy ...
print(testB.bar())  # implementácia z triedy ...

