# Tabu Search implementation for Travelling Salesman Problem
# test: python tabu_search.py -f data.txt -i numberOfIterations -s sizeOfTabuSearch

import argparse
import copy
import random


def generate_neighbours(path):
    # створюємо словник для сусідів
    dict_of_neighbours = {}
    # начитуємо дані з файлу
    with open(path) as data_file:
        # зчитуємо рядки
        content = data_file.readlines()
        # розбиваємо рядки на початковий вузол кінцевий та відстань між ними(починаємо з 2 рядку тому що перший це список всіх вузлів)
        for i in range(1, len(content)):
            start_node = content[i].split()[0]
            finish_node = content[i].split()[1]
            distance = content[i].split()[2]
            # перевірка на наявність вузлів у словнику
            if start_node not in dict_of_neighbours:
                # записуємо в словник сусідній вузол та відстань між ними
                temp = [[finish_node, distance]]
                dict_of_neighbours[start_node] = temp
            # якщо вузол вже є в словнику то додаємо по ключу(вузлу який є у словнику) сусідній вузол та відстань між ними
            else:
                dict_of_neighbours[start_node].append([finish_node, distance])
            # перевірка на наявність кінцевого вузлу
            if finish_node not in dict_of_neighbours:
                # якщо його немає в словнику то додаємо кінцевий вузол з параметрами початковий вузол та дистанція між ними
                temp = [[start_node, distance]]
                dict_of_neighbours[finish_node] = temp
            # якщо вузол вже є в словнику то додаємо по ключу(вузлу який є у словнику) сусідній вузол та відстань між ними
            else:
                dict_of_neighbours[finish_node].append([start_node, distance])
    return dict_of_neighbours

def generate_initial_solution(path, dict_of_neighbours): #add node destination node
    # variables
    # remake for specific node
    # начитаємо преший рядок з файлу та розділимо його, за пробілом між ними, для вибору вузла
    with open(path) as data_file:
        start_line = data_file.readline().split(' ')
    # rewrite for user input
    start_node = input(f"Nodes that you can choose: {start_line[:-1]}\n>>>: ")
    # присвоюємо кінцевому вузлу значення початкового оскільки потрібно зробити коло і повернутися в початковий вузол
    finish_node = start_node
    initial_solution = []
    current_node = start_node
    total_distance = 0

    # creation of initial solution
    while current_node not in initial_solution:
        min_distance = 10000
        for k in dict_of_neighbours[current_node]:
            current_distance = k[1]
            current_neighbour = k[0]
            # порівнюємо відстань та перевіряємо наявнясть сусіда
            if int(current_distance) < int(min_distance) and current_neighbour not in initial_solution:
                min_distance = current_distance
                nearest_node = current_neighbour
        # додаємо вузол до початкового рішення, підраховуємо дистанцію
        initial_solution.append(current_node)
        total_distance += int(min_distance)
        current_node = nearest_node

    initial_solution.append(finish_node)

    # calculate distance from second to last element to last one
    
    # єбаторія для обрахування відстані між перед останнім та останнім вузлом
    position = 0
    for k in dict_of_neighbours[initial_solution[-2]]:
        if k[0] == start_node:
            break
        position += 1
        
    # Calculate the distance between the penultimate and the last node to complete the cycle
    if len(initial_solution) > 1:
        last_node = initial_solution[-2]
        start_node_distance = next((int(n[1]) for n in dict_of_neighbours[last_node] if n[0] == start_node), None)
        if start_node_distance:
            total_distance += start_node_distance
            total_distance -= 10000
    else:
        total_distance = 0
    
    # return initial solution and it's distance
    return initial_solution, total_distance


def find_neighborhood(current_solution, neighbors_dict):
    # The function implementation remains the same as previously defined.
    neighborhood = []
    # беремо від першого по останній елемент(не включно останні) оскільки у нас коло то перший і останній елементи однакові
    for node in current_solution[1:-1]:
        original_index = current_solution.index(node)
        # --||--
        for neighbor_node in current_solution[1:-1]:
            # перевірка аби ми не міняли однакові вузли між собою
            if node == neighbor_node:
                continue
            # змінюємо елементи між собою створюючи нове рішення
            temp_solution = copy.deepcopy(current_solution)
            temp_solution[original_index] = neighbor_node
            temp_solution[current_solution.index(neighbor_node)] = node

            total_distance = 0
            # цей весь блок для оцінки рішення
            for i in range(len(temp_solution) - 1):
                current_node = temp_solution[i]
                next_node = temp_solution[i + 1]
                
                for neighbor in neighbors_dict[current_node]:
                    if neighbor[0] == next_node:
                        total_distance += int(neighbor[1])
            
            temp_solution.append(total_distance)
            # якщо рішення НЕ присутнє то ми його додаємо в список решень
            if temp_solution not in neighborhood:
                neighborhood.append(temp_solution)
    # сортуємо по останньму елементу
    neighborhood.sort(key=lambda x: x[-1])
    return neighborhood


def tabu_search(
    first_solution, distance_of_first_solution, dict_of_neighbours, iters, size
):
    # Лічильник ітерацій починається з 1, оскільки нульова ітерація - це початкове рішення
    count = 1
    initial_solution = first_solution
    tabu_list = []
    best_cost = distance_of_first_solution
    best_solution_ever = initial_solution

    # Виконуємо ітерації, поки не досягнемо заданої кількості
    while count <= iters:
        # Генеруємо можливі комбінації
        neighborhood = find_neighborhood(initial_solution, dict_of_neighbours)
        # починаємо з нуля тому що список рішень відсортований і "найкраще" рішення буде першим
        index_of_best_solution = 0
        best_solution = neighborhood[index_of_best_solution]
        # Індекс для доступу до загальної вартості найкращого рішення
        best_cost_index = len(best_solution) - 1

        # Припускаєм що у нас немає кращого рішення
        # Прапорець для того аби в циклі знайти якого ще немає в табу-листі
        found = False
        # Поки не знайдемо рішення, яке не входить до Табу-листа
        while not found:
            i = 0
            # Шукаємо першу відмінність між поточним та найкращим рішенням
            while i < len(best_solution):
                if best_solution[i] != initial_solution[i]:
                    first_exchange_node = best_solution[i]
                    second_exchange_node = initial_solution[i]
                    break
                i = i + 1

            # Перевіряємо, чи не входить пара до табу листа
            if [first_exchange_node, second_exchange_node] not in tabu_list and [
                second_exchange_node,
                first_exchange_node,
            ] not in tabu_list:
                # Додаємо обмін до Табу-листа
                tabu_list.append([first_exchange_node, second_exchange_node])
                # встановлюємо прапорець на ТРУ, так як додали рішення до табу листа
                found = True
                # Оновлюємо поточне рішення та прибираємо відстань
                initial_solution = best_solution[:-1]
                # Визначаємо вартість найкращого сусіднього рішення
                cost = neighborhood[index_of_best_solution][best_cost_index]
                # Якщо нова вартість менша за поточну найкращу, оновлюємо найкраще рішення
                if cost < best_cost:
                    best_cost = cost
                    best_solution_ever = initial_solution
            else:
                # Переходимо до наступного найкращого рішення серед сусідів
                index_of_best_solution = index_of_best_solution + 1
                best_solution = neighborhood[index_of_best_solution]

        # Якщо Табу-лист перевищує заданий розмір, видаляємо найстаріший елемент
        if len(tabu_list) >= size:
            tabu_list.pop(0)

        # Збільшуємо лічильник ітерацій
        count = count + 1

    # Повертаємо найкраще знайдене рішення за весь час та його загальну вартість
    return best_solution_ever, best_cost



def main(args=None):
    dict_of_neighbours = generate_neighbours(args.File)

    first_solution, distance_of_first_solution = generate_initial_solution(
        args.File, dict_of_neighbours
    )

    best_sol, best_cost = tabu_search(
        first_solution,
        distance_of_first_solution,
        dict_of_neighbours,
        args.Iterations,
        args.Size,
    )

    print(f"Best solution: {best_sol}, with total distance: {best_cost}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tabu Search")
    parser.add_argument(
        "-f",
        "--File",
        type=str,
        help="Path to the file containing the data",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--Iterations",
        type=int,
        help="How many iterations the algorithm should perform",
        required=True,
    )
    parser.add_argument(
        "-s", "--Size", type=int, help="Size of the tabu list", required=True
    )
    
    # Pass the arguments to main method
    main(parser.parse_args())

    # neighborhood = generate_neighbours("C:/Users/denys/Desktop/HOP/tabu_test_data.txt")
    # print(neighborhood)
    # start_sol, dist = generate_initial_solution("C:/Users/denys/Desktop/HOP/tabu_test_data.txt", neighborhood)
    # print(start_sol)
    # print(dist)
    
    # neighbors_dict = {
    #     'a': [['b', '20'], ['c', '18'], ['d', '22'], ['e', '26']],
    #     'c': [['a', '18'], ['b', '10'], ['d', '23'], ['e', '24']],
    #     'b': [['a', '20'], ['c', '10'], ['d', '11'], ['e', '12']],
    #     'e': [['a', '26'], ['b', '12'], ['c', '24'], ['d', '40']],
    #     'd': [['a', '22'], ['b', '11'], ['c', '23'], ['e', '40']]
    # }

    # current_solution = ['a', 'c', 'b', 'd', 'e', 'a']
    # expected_output = [
    #     ['a', 'e', 'b', 'd', 'c', 'a', 90],
    #     ['a', 'c', 'd', 'b', 'e', 'a', 90],
    #     ['a', 'd', 'b', 'c', 'e', 'a', 93],
    #     ['a', 'c', 'b', 'e', 'd', 'a', 102],
    #     ['a', 'c', 'e', 'd', 'b', 'a', 113],
    #     ['a', 'b', 'c', 'd', 'e', 'a', 119]
    # ]

    # result = find_neighborhood(current_solution, neighbors_dict)
    
    # print("Result:")
    # for r in result:
    #     print(r)

    # assert result == expected_output, "Test failed!"
    # print("Test passed!")

