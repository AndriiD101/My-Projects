# Meno: Denysenko, Andrii
# Spolupráca: 
# Použité zdroje: 
# Čas: 

# Podrobný popis je dostupný na: https://github.com/ianmagyar/introduction-to-python/blob/master/assignments/homeworks/homework05.md

# Hodnotenie: /1b

# 24
def concatenate_unique_elements(list1, list2):
    if len(list1) == 0 and len(list2) == 0:
        raise ValueError("both list must contain at least one element")
    if len(list1) == 0:
        raise ValueError("first list must contain at least one element")
    if len(list2) == 0:
        raise ValueError("second list must contain at least one element")
    if not isinstance(list1, list) or not isinstance(list2, list):
        raise ValueError("Both inputs must be lists")
    # Function concatenates unique elements from two lists
    result_list = list1[:]  # Start with a copy of the first list to avoid altering it
    for element in list2:
        if element not in result_list:
            result_list.append(element)
    return result_list


def test_concatenate_unique_elements():

    simple_test_cases = [
        ([1, 2, 3], [3, 4, 5], [1, 2, 3, 4, 5]),
        ([1, 2, 3], [3, 4, 5, 1], [1, 2, 3, 4, 5]),
        (["a", "b", "c"], ["d", "e", "f"], ["a", "b", "c", "d", "e", "f"]),
        (["apple", "banana", "cherry"], ["banana", "orange", "cherry"], ["apple", "banana", "cherry", "orange"]),
        (["a", 1, "b", 2], [1, 2, "c", "d"], ["a", 1, "b", 2, "c", "d"]),
        ([[12, 56, 234], [65, 87, 43]], [[12, 56, 234], [76, 9845, 294]], [[12, 56, 234], [65, 87, 43], [76, 9845, 294]]),
        ([1, 2, 3], [2, 3, 1], [1, 2, 3])
    ]
    border_tests =[
        (list(range(10**4)), list(range(10**4, 2 * 10**4)), list(range(2 * 10**4))),
        ([0] * 10**4, [0] * 10**4, [0] * 10**4),
    ]
    border_tests2 =[
        ([], []), # I don't know if this 3(2 below) inputs are extremely, I guess this should be just as I have written
        ([1, 2, 3], []),#This
        ([], [4, 5, 6]),#and also this
    ]
    for index, (elm1, elm2, expected_result) in enumerate(simple_test_cases):
        result = concatenate_unique_elements(elm1, elm2)
        assert result == expected_result, f"Test case {index + 1} failed: Expected {expected_result}, but got {result}"
    print("programm passed simple tests successfully")
    
    for index, (elm1, elm2, expected_result) in enumerate(border_tests):
        result = concatenate_unique_elements(elm1, elm2)
        assert result == expected_result, f"Test case {index + 1} failed: Expected {expected_result}, but got {result}"
    print("programm passed complex tests successfully")
    
    try:
        print(concatenate_unique_elements({'a': 1, 'b': 2}, "odfsosndfsodfu"))
    except Exception as error:
        print(f"Some problem occurred during running test: {error}")

    try:
        print(concatenate_unique_elements(123, [1, 2, 3]))
    except Exception as error:
        print(f"Some problem occurred during running test: {error}")
    for list1, list2 in border_tests2:
        try:
            print(concatenate_unique_elements(list1, list2))
        except Exception as error:
            print(f"Some problem occurred during running test: {error}")


test_concatenate_unique_elements()
