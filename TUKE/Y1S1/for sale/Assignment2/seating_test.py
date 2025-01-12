from copy import deepcopy
import pickle
import random

from employee import Employee
import seating


def test_load_example(verbose=False):
    print("Testing seating.load_example():")
    points = 0

    for test_no in range(1, 21):
        src_file = "samples/test_{}.csv".format(test_no)
        with open(src_file, 'r') as src:
            file_lines = src.readlines()

        try:
            st_res = seating.load_example(src_file)
        except Exception as e:
            if verbose:
                print(f"\tCalling seating.load_example() caused an error (test case {test_no}:")
                print("\t", e)
            continue
        else:
            if not isinstance(st_res, list):
                if verbose:
                    print(f"\tseating.load_example() returned wrong value type. Expected list, got {type(st_res)}")
                continue
            elif len(st_res) != len(file_lines):
                if verbose:
                    print(f"\tseating.load_example() returned wrong number of employees. Expected {len(file_lines)}, got{len(st_res)}")
                continue
            else:
                for emp_idx, (line, emp) in enumerate(zip(file_lines, st_res)):
                    if not isinstance(emp, Employee):
                        if verbose:
                            print(f"\tseating.load_example() should return a list of Employees, {type(emp)} found")
                        continue

                    # check drives, seating, contancts
                    drives, front, contacts = line.split(',', maxsplit=2)

                    if str(emp.will_drive) != drives:
                        if verbose:
                            print(f"\tEmployee {emp_idx} should have a will_drive value of {drives}, {emp.will_drive} found")
                        continue

                    if str(emp.sits_in_front) != front:
                        if verbose:
                            print(f"\tEmployee {emp_idx} should have a sits_in_front value of {front}, {emp.sits_in_front} found")
                        continue

                    contact_pairs = contacts[:-1].split(',')
                    if len(emp.contacts) != len(contact_pairs):
                        if verbose:
                            print(f"\tEmployee {emp_idx} has incorrect number of contacts. {len(contact_pairs)} expected, {len(emp.contacts)} found")
                        continue

                    has_contacts = True
                    for contact_info in contact_pairs:
                        emp_id, weight = contact_info.split(':')
                        contact_emp = st_res[int(emp_id)]

                        try:
                            contact_weight = emp.contacts[contact_emp]
                        except KeyError:
                            if verbose:
                                print(f"\tMissing contact for employee {emp_idx}: {emp_id}")
                            has_contacts = False
                            continue
                        else:
                            if contact_weight != float(weight):
                                if verbose:
                                    print(f"\tContact {emp_id} for employee {emp_idx} has wrong weight. Expected {weight}, found {contact_weight}")
                                has_contacts = False
                                continue

                    if not has_contacts:
                        continue

                points += 0.1

    print("Testing seating.load_example() finished: {:.2f}/2 points".format(points))
    return points


def build_employee_list(employees):
    employee_list = []
    for drives, front, _ in employees:
        employee_list.append(Employee(drives, front))

    for idx, (_, _, contacts) in enumerate(employees):
        employee_list[idx].contacts = dict()
        for cont_id, weight in contacts:
            employee_list[idx].contacts[employee_list[cont_id]] = weight

    return employee_list


def test_get_avg_satisfaction(verbose=False):
    print("Testing seating.get_avg_satisfaction():")
    points = 0

    with open("avg_sat_cases.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for test_idx, (trans_emp_list, car_count, possibles) in enumerate(test_cases):
        emp_list = build_employee_list(trans_emp_list)

        try:
            st_res = seating.get_avg_satisfaction(emp_list, car_count)
        except Exception as e:
            if verbose:
                print(f"\tCalling seating.get_avg_satisfaction() caused an error (test case {test_idx}:")
                print("\t", e)
            continue
        else:
            if not isinstance(st_res, float):
                if verbose:
                    print(f"\tseating.get_avg_satisfaction() returned a value of the wrong type. Expected float, got {type(st_res)}")
                continue
            elif st_res not in possibles:
                if verbose:
                    print(f"\tseating.get_avg_satisfaction() returned wrong value: {st_res}, accepted: {possibles}")
                continue
            else:
                correct += 1

    points = (correct / all_tests) * 0.5
    print("Testing seating.get_avg_satisfaction() finished: {:.2f}/0.5 points".format(points))
    return points


def test_get_all_seatings(verbose=True):
    print("Testing seating.get_all_seatings():")
    points = 0

    types_ok = True
    lengths_fine = True
    always_empty = True

    with open("seatings.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
        # print (test_cases)
    correct = 0
    all_tests = len(test_cases)

    for test_idx, (inp, corr, _) in enumerate(test_cases):
        emp_list = build_employee_list(inp)
        car_count = len(emp_list) // 4
        # print(inp)
        # print(corr)
        try:
            st_res = seating.get_all_seatings(emp_list, car_count)
        except Exception as e:
            if verbose:
                print(f"\tCalling seating.get_all_seatings() caused an error (test case {test_idx}:")
                print("\t", e)
            continue
        else:
            if not isinstance(st_res, list):
                if verbose:
                    print(f"\tseating.get_all_seatings() returned wrong type. Expected list, got {type(st_res)}")
                types_ok = False
                continue
            else:
                for poss_seating in st_res:
                    if not isinstance(poss_seating, list):
                        if verbose:
                            print(f"\tseating.get_all_seatings() returned wrong type. Expected list of lists, found {type(poss_seating)}")
                        types_ok = False
                        continue
                    else:
                        for car in poss_seating:
                            if not isinstance(car, list):
                                if verbose:
                                    print(f"\tseating.get_all_seatings() returned wrong type. Expected list of list of lists, found {type(car)}")
                                types_ok = False
                                continue
                            else:
                                for emp in car:
                                    if not isinstance(car, list):
                                        if verbose:
                                            print(f"\tseating.get_all_seatings() returned wrong type. Expected employees in car, found {type(emp)}")
                                        types_ok = False
                                        continue
                else:
                    if len(st_res) != 0:
                        always_empty = False
                    if len(st_res) < len(corr):
                        if verbose:
                            print(f"\tseating.get_all_seatings() returned too few seatings. Expected {len(corr)}, got {len(st_res)}")
                        lengths_fine = False
                        continue
                    else:
                        test_pass = True
                        for poss_seating in st_res:
                            if st_res.count(poss_seating) != 1:
                                if verbose:
                                    print(f"\tduplicate found in seating.get_all_seatings()")
                                test_pass = False
                                break

                            # look for non-exact duplicates
                            test_seating = deepcopy(poss_seating)
                            while test_seating == poss_seating:
                                for car_idx, car in enumerate(test_seating):
                                    if random.random() < 0.5:
                                        test_seating[car_idx] = [car[0], car[1], car[3], car[2]]
                            if test_seating in st_res:
                                if verbose:
                                    print(f"\tduplicate found in seating.get_all_seatings(): {poss_seating} and {test_seating}")
                                test_pass = False
                                break

                        if test_pass:
                            correct += 1

    points = correct / all_tests
    if types_ok:
        points += 0.5
    if lengths_fine:
        points += 0.5
    if always_empty:
        points = 0.0
    print("Testing seating.get_all_seatings() finished: {:.2f}/2 points".format(points))
    return points


def test_get_optimal_satisfaction(verbose=False):
    print("Testing seating.get_optimal_satisfaction():")
    points = 0

    with open("seatings.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for test_idx, (inp, _, avg_sats) in enumerate(test_cases):
        emp_list = build_employee_list(inp)
        car_count = len(emp_list) // 4

        try:
            st_res = seating.get_optimal_satisfaction(emp_list, car_count)
        except Exception as e:
            if verbose:
                print(f"\tCalling seating.get_optimal_satisfaction() caused an error (test case {test_idx}:")
                print("\t", e)
            continue
        else:
            if not isinstance(st_res, float):
                if verbose:
                    print(f"\tseating.get_optimal_satisfaction() returned wrong value type. Expected float, got {type(st_res)}")
                continue
            else:
                true_max = max(avg_sats)
                if abs(true_max - st_res) > 1e-4:
                    if verbose:
                        print(f"\tseating.get_optimal_satisfaction() returned wrong value. Expected {true_max}, got {st_res}")
                    continue
                else:
                    correct += 1

    points = (correct / all_tests) * 0.5
    print("Testing seating.get_optimal_satisfaction() finished: {:.2f}/0.5 points".format(points))
    return points


def test_get_seating_order(verbose=False):
    print("Testing seating.get_seating_order():")
    points = 0

    with open("seatings.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)

    types_ok = True

    for test_idx, (inp, _, _) in enumerate(test_cases[:6]):
        emp_list = build_employee_list(inp)
        car_count = len(emp_list) // 4

        try:
            st_res = seating.get_seating_order(deepcopy(emp_list), car_count)
        except Exception as e:
            if verbose:
                print(f"\tCalling seating.get_seating_order() caused an error (test case {test_idx}:")
                print("\t", e)
            continue
        else:
            if not isinstance(st_res, list):
                if verbose:
                    print(f"\tseating.get_seating_order() returned wrong value type. Expected list, got {type(st_res)}")
                types_ok = False
                continue
            else:
                for elem in st_res:
                    if not isinstance(elem, Employee):
                        if verbose:
                            print(f"\tseating.get_seating_order() returned wrong value type. Expected list of Employees, found {type(elem)}")
                        types_ok = False
                        continue
                else:
                    all_good = True

                    scores = [seating.get_avg_satisfaction(deepcopy(st_res), car_count) for _ in range(20)]
                    best_score = max(scores)
                    for _ in range(100):
                        test_list = deepcopy(emp_list)
                        random.shuffle(test_list)
                        try:
                            score = seating.get_avg_satisfaction(test_list, car_count)
                        except Exception:
                            pass
                        else:
                            if score > best_score and abs(score - best_score) > 1e-4:
                                if verbose:
                                    print(f"\tseating.get_seating_order() returned wrong value. Better combination found with score {score} compared to {best_score}")
                                all_good = False

                    if all_good:
                        points += 0.15

    if types_ok:
        points += 0.1
    print("Testing seating.get_seating_order() finished: {:.2f}/1 point".format(points))
    return points


def main():
    total = 0
    # total += test_load_example()

    # total += test_get_avg_satisfaction()

    total += test_get_all_seatings()

    # total += test_get_optimal_satisfaction()

    # total += test_get_seating_order()

    print()
    print("EMPLOYEE: {:.2f}/6 points".format(total))


if __name__ == '__main__':
    main()


