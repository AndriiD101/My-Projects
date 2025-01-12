# -*- coding: utf-8 -*-

# Meno: Denysenko, Andrii
# Spolupráca: 
# Použité zdroje: 
# Čas: 

# Podrobný popis je dostupný na: https://github.com/ianmagyar/introduction-to-python/blob/master/assignments/homeworks/homework09.md

# Hodnotenie: /1b

import numpy
import pandas

# --------------------
# Úloha 1
# Načítajte dataset uložený v súbore h03.csv ako pandas DataFrame
# a určte pomocou metód pandas dataframe (alebo cez použitie numpy poľa):
#  - priemernú výšku mužov

df = pandas.read_csv('h09.csv')
average_male_height = df[df['gender'] == 'male']['height'].mean()
print(f'Average male height: {average_male_height}')

# --------------------
# Úloha 2
# V kóde nižšie sa vygeneruje dvojrozmerné numpy pole s náhodnými číselnými
# hodnotami. Vypočítajte:
#  - strednú hodnotu (medián) po stĺpcoch
array = numpy.random.rand(5, 5)
print(array)

median_per_column = numpy.median(array, axis=0)
print(f'Median per column: {median_per_column}')
