# Meno: Denysenko, Andrii
# Spolupráca: 
# Použité zdroje: 
# Čas: 

# Podrobný popis je dostupný na: https://github.com/ianmagyar/introduction-to-python/blob/master/assignments/homeworks/homework04.md

# Hodnotenie: /1b

# 28
from collections.abc import Iterable

def get_largest(lst):
    # returns the largest number in a list
    # lst should be a list of numbers
    try:
        return max(lst)
    except TypeError as a:
        print(f"TypeError: {a}")
    except ValueError as e:
        print(f"ValueError: {e}")

    # TODO: check the validity of input, amend optimistic code
    # e.g. is lst a list? does it contain numbers only?

# def remove_key(dct, key):
#     if not isinstance(dct, dict):
#         print(">> Error. Function argument 'dct' is of the wrong type")
#     if not isinstance(key, str):
#         print(">> Error. Function argument 'key' is of the wrong type")
#     if len(dct) == 0:
#         print(">> Error. 'dct' argument is an empty dictionary.")
#     if len(key) == 0:
#         print(">> Error. 'key' argument is an empty string.")
#     try:
#         del dct[key]
#     except KeyError:
#         print(">> Error. Value of 'key' argument is not in 'dct'.")

