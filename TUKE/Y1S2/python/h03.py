# Meno: Denysenko, Andrii
# Spolupráca: 
# Použité zdroje: 
# Čas: 14.31 02.03.24 

# Podrobný popis je dostupný na: https://github.com/ianmagyar/introduction-to-python/blob/master/assignments/homeworks/homework03.md

# Hodnotenie: /1b

import random
import string

# --------------------
# Úloha 1

# 17. Máme zoznam reťazcov lst. Pomocou list comprehension vygenerujte zoznam lst2,
# ktorý bude obsahovať iba tie prvky zoznamu lst, ktoré začínajú na písmeno s.
lst = [''.join(random.choices(string.ascii_lowercase, k=6)) for _ in range(20)]
print(lst)
# lst = ["string", "fdlfjdlfj", "othbnn", "sotyubon"]
lst2 = []
for i in range(len(lst)):
    if lst[i][0] == "s":
        lst2.append(lst[i])
print(lst2)


# 3. Máme zoznam čísel lst. Pomocou list comprehension vygenerujte zoznam lst2,
# ktorý bude obsahovať iba dvojciferné prvky zoznamu lst.
lst = [random.randint(1, 1000) for _ in range(20)]
print(lst)
lst2 = []
for i in lst:
    if 10<=i<100:
        lst2.append(i)
print(lst2)


# --------------------
# Úloha 2

# 31. Máme zoznam čísel lst. Upravte zoznam pomocou lambda výrazu tak,
# že z jednotlivých čísel vypočítate ich tretiu mocninu.
lst = [random.randint(1, 1000) for _ in range(20)]
print(lst)
edited_list = list(map(lambda x: x ** 3, lst))
print(edited_list)
