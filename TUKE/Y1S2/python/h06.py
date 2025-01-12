# Meno: Denysenko, Andrii
# Spolupráca: 
# Použité zdroje: 
# Čas: 

# Podrobný popis je dostupný na: https://github.com/ianmagyar/introduction-to-python/blob/master/assignments/homeworks/homework06.md

# Hodnotenie: /1b

# TODO: add your definition of the class

class Striker:
    
    position_code = 0
    
    def __init__(self, name: str, jersey: int):
        self.name = name
        self.__jersey = jersey
        Striker.position_code += 1
    
    def get_jersey(self)->int:
        return self.__jersey
    
    def __str__(self) -> str:
        return "Striker {} {}".format(self.name, self.__jersey)