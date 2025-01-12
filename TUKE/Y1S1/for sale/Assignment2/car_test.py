import pickle
import random

from car import Car
from employee import Employee


def test_constructor(verbose=False):
    print("Testing Car constructor and structure:")

    points = 0
    try:
        test_car = Car()
    except Exception as e:
        if verbose:
            print(f"\tException occurred when calling Car constructor: {e}")
    else:
        try:
            seats_list = test_car.seats
        except Exception:
            if verbose:
                print("\tCannot find seats attribute in car")
        else:
            if not isinstance(seats_list, list):
                if verbose:
                    print(f"\tCar.seats attribute is of wrong type. Expected list, found {type(seats_list)}")
            elif len(seats_list) != 4:
                if verbose:
                    print(f"\tCar.seats attribute is of wrong length. Expected 4, found {len(seats_list)}")
            elif seats_list != [None, None, None, None]:
                if verbose:
                    print(f"\tCar.seats attribute has wrong value after initialization. Expected [None, None, None, None], found {seats_list}")
            else:
                points = 0.1

    print("Testing Car constructor finished: {:.2f}/0.1 points".format(points))
    return points


def test_has_driver(verbose=False):
    print("Testing Car.has_driver():")
    points = 0

    with open("car_examples.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (seats_list, has_driver, _, _, _, _, _) in enumerate(test_cases):
        try:
            test_car = Car()
            test_car.seats = seats_list
            st_res = test_car.has_driver()
        except Exception as e:
            if verbose:
                print(f"\tAn exception occurred when calling Car.has_driver() (test case {idx + 1}):")
                print("\t", e)
        else:
            if not isinstance(st_res, bool):
                if verbose:
                    print(f"\tCar.has_driver() returned wrong type. Expected bool, got {type(st_res)}")
                continue

            if st_res == has_driver:
                correct += 1
            elif verbose:
                print(f"\tCar.has_driver() returned wrong value (test case {idx + 1})")
                print(f"\tExpected {has_driver} for {seats_list}, got {st_res}")

    points = (correct / all_tests) * 0.1
    print("Testing Car.has_driver() finished: {:.2f}/0.1 points".format(points))
    return points


def test_has_empty_front(verbose=False):
    print("Testing Car.has_empty_front():")
    points = 0

    with open("car_examples.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (seats_list, _, has_empty_front, _, _, _, _) in enumerate(test_cases):
        try:
            test_car = Car()
            test_car.seats = seats_list
            st_res = test_car.has_empty_front()
        except Exception as e:
            if verbose:
                print(f"\tAn exception occurred when calling Car.has_empty_front() (test case {idx + 1}):")
                print("\t", e)
        else:
            if not isinstance(st_res, bool):
                if verbose:
                    print(f"\tCar.has_empty_front() returned wrong type. Expected bool, got {type(st_res)}")
                continue

            if st_res == has_empty_front:
                correct += 1
            elif verbose:
                print(f"\tCar.has_empty_front() returned wrong value (test case {idx + 1})")
                print(f"\tExpected {has_empty_front} for {seats_list}, got {st_res}")

    points = (correct / all_tests) * 0.1
    print("Testing Car.has_empty_front() finished: {:.2f}/0.1 points".format(points))
    return points


def test_has_empty_back(verbose=False):
    print("Testing Car.has_empty_back():")
    points = 0

    with open("car_examples.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (seats_list, _, _, has_empty_back, _, _, _) in enumerate(test_cases):
        try:
            test_car = Car()
            test_car.seats = seats_list
            st_res = test_car.has_empty_back()
        except Exception as e:
            if verbose:
                print(f"\tAn exception occurred when calling Car.has_empty_back() (test case {idx + 1}):")
                print("\t", e)
        else:
            if not isinstance(st_res, bool):
                if verbose:
                    print(f"\tCar.has_empty_back() returned wrong type. Expected bool, got {type(st_res)}")
                continue

            if st_res == has_empty_back:
                correct += 1
            elif verbose:
                print(f"\tCar.has_empty_back() returned wrong value (test case {idx + 1})")
                print(f"\tExpected {has_empty_back} for {seats_list}, got {st_res}")

    points = (correct / all_tests) * 0.1
    print("Testing Car.has_empty_back() finished: {:.2f}/0.1 points".format(points))
    return points


def test_get_number_of_empty(verbose=False):
    print("Testing Car.get_number_of_empty():")
    points = 0

    with open("car_examples.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (seats_list, _, _, _, no_empty, _, _) in enumerate(test_cases):
        try:
            test_car = Car()
            test_car.seats = seats_list
            st_res = test_car.get_number_of_empty()
        except Exception as e:
            if verbose:
                print(f"\tAn exception occurred when calling Car.get_number_of_empty() (test case {idx + 1}):")
                print("\t", e)
        else:
            if not isinstance(st_res, int):
                if verbose:
                    print(f"\tCar.get_number_of_empty() returned wrong type. Expected int, got {type(st_res)}")
                continue

            if st_res == no_empty:
                correct += 1
            elif verbose:
                print(f"\tCar.get_number_of_empty() returned wrong value (test case {idx + 1})")
                print(f"\tExpected {no_empty} for {seats_list}, got {st_res}")

    points = (correct / all_tests) * 0.1
    print("Testing Car.get_number_of_empty() finished: {:.2f}/0.1 points".format(points))
    return points


def test_add_passenger(verbose=False):
    print("Testing Car.add_passenger():")
    points = 0

    with open("car_examples.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = 0

    # TODO: test for TypeError when dummy not employee

    # test for invalid indeces
    all_good = True
    for test_val in [-4, -3, -2, -1, 4, 5, 6, 7]:
        dummy = Employee(driver=True, front_seater=True)
        try:
            test_car = Car()
            test_car.add_passenger(dummy, test_val)
        except ValueError as e:
            if str(e) == f"Cannot add passenger at index {test_val}":
                pass
            elif verbose:
                print(f"\tCar.add_passenger() raised ValueError with wrong message (test case {idx + 1}):")
                print("\t", e)
                all_good = False
                break
        except Exception as e:
            if verbose:
                print(f"\tCar.add_passenger() raised wrong error for invalid index (test case {idx + 1}):")
                print("\t", e)
                all_good = False
                break
        else:
            if verbose:
                print(f"\tCar.add_passenger() did not raise an error for invalid index (test case {idx + 1})")
                print(f"\tExpected ValueError for {seats_list}, index {test_val}")
                all_good = False
                break
    if all_good:
        points += 0.1

    # test for invalid additions
    for idx, (seats_list, _, _, _, _, empties, _) in enumerate(test_cases[:50]):
        if len(empties) == 0:
            continue
        test_val = empties[0]
        while test_val in empties:
            test_val = random.randint(0, 3)
        dummy = Employee(driver=True, front_seater=True)
        all_tests += 1
        try:
            test_car = Car()
            test_car.seats = seats_list
            test_car.add_passenger(dummy, test_val)
        except ValueError as e:
            if str(e) == f"Position {test_val} already occupied":
                correct += 1
            elif verbose:
                print(f"\tCar.add_passenger() raised ValueError with wrong message (test case {idx + 1}):")
                print("\t", e)
        except Exception as e:
            if verbose:
                print(f"\tCar.add_passenger() raised wrong error for invalid addition (test case {idx + 1}):")
                print("\t", e)
        else:
            if verbose:
                print(f"\tCar.add_passenger() did not raise an error for invalid addition (test case {idx + 1})")
                print(f"\tExpected ValueError for {seats_list}, index {test_val}")

    # test for valid additions
    for idx, (seats_list, _, _, _, _, empties, _) in enumerate(test_cases[50:]):
        if len(empties) == 0:
            continue
        test_val = random.choice(empties)
        dummy = Employee(driver=True, front_seater=True)
        all_tests += 1
        try:
            test_car = Car()
            test_car.seats = seats_list
            test_car.add_passenger(dummy, test_val)
        except Exception as e:
            if verbose:
                print(f"\tAn exception occurred when calling Car.add_passenger() (test case {idx + 1}):")
                print("\t", e)
        else:
            if test_car.seats[test_val] == dummy:
                correct += 1
            elif verbose:
                print(f"\tCar.add_passenger() did not add passenger to empty seat (test case {idx + 1})")
                print(f"\tExpected success for {seats_list}, index {test_val}")


    points += (correct / all_tests) * 0.2
    print("Testing Car.add_passenger() finished: {:.2f}/0.3 points".format(points))
    return points


def test_get_car_satisfaction(verbose=False):
    print("Testing Car.get_car_satisfaction():")
    points = 0

    with open("car_examples.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (seats_list, _, _, _, _, _, corr_sat) in enumerate(test_cases):
        try:
            test_car = Car()
            test_car.seats = seats_list
            st_res = test_car.get_car_satisfaction()
        except Exception as e:
            if verbose:
                print(f"\tAn exception occurred when calling Car.get_car_satisfaction() (test case {idx + 1}):")
                print("\t", e)
        else:
            if not isinstance(st_res, float):
                if verbose:
                    print(f"\tCar.get_car_satisfaction() returned wrong type. Expected float, got {type(st_res)}")
                continue

            if abs(st_res - corr_sat) < 1e-6:
                correct += 1
            elif verbose:
                print(f"\tCar.get_car_satisfaction() returned wrong value (test case {idx + 1})")
                print(f"\tExpected {corr_sat} for, got {st_res}")


    points = (correct / all_tests) * 0.2
    print("Testing Car.get_car_satisfaction() finished: {:.2f}/0.2 points".format(points))
    return points


def main():
    total = 0
    total += test_constructor()

    total += test_has_driver()

    total += test_has_empty_front()

    total += test_has_empty_back()

    total += test_get_number_of_empty()

    total += test_add_passenger()

    total += test_get_car_satisfaction()

    print()
    print("CAR: {:.2f}/1 point".format(total))


if __name__ == '__main__':
    main()
