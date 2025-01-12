import pickle
import random

from car import Car
from employee import Employee


def test_constructor(verbose=False):
    print("Testing Employee constructor and structure:")

    has_error = False
    try:
        test_employee = Employee(driver=True, front_seater=True)
    except Exception as e:
        if verbose:
            print(f"\tException occurred when calling Employee constructor: {e}")
    else:
        try:
            will_drive = test_employee.will_drive
        except Exception:
            if verbose:
                print("\tCannot find will_drive attribute in employee")
        else:
            if not isinstance(will_drive, bool):
                if verbose:
                    print(f"\tEmployee.will_drive attribute is of wrong type. Expected bool, found {type(will_drive)}")
                has_error = True

        try:
            sits_in_front = test_employee.sits_in_front
        except Exception:
            if verbose:
                print("\tCannot find sits_in_front attribute in employee")
        else:
            if not isinstance(sits_in_front, bool):
                if verbose:
                    print(f"\tEmployee.sits_in_front attribute is of wrong type. Expected bool, found {type(sits_in_front)}")
                has_error = True

        try:
            contacts = test_employee.contacts
        except Exception:
            if verbose:
                print("\tCannot find contacts attribute in employee")
        else:
            if not isinstance(contacts, dict):
                if verbose:
                    print(f"\tEmployee.contacts attribute is of wrong type. Expected bool, found {type(contacts)}")
                has_error = True
            elif len(contacts) != 0:
                if verbose:
                    print(f"\tEmployee.contacts attribute has wrong value. Expected empty dictionary, found {len(contacts)} elements.")
                has_error = True

    points = 0.0 if has_error else 0.25
    print("Testing Employee constructor finished: {:.2f}/0.25 points".format(points))
    return points


def test_set_contacts(verbose=False):
    print("Testing Employee.set_contacts():")
    points = 0
    generates_type_err = True
    generates_value_err = True

    try:
        test_employee = Employee(driver=True, front_seater=True)
        emp2 = Employee(driver=True, front_seater=True)
        test_employee.set_contacts([(emp2, 0.5), (('bla', 'bla'), 0.1)])
    except TypeError as e:
        pass
    except Exception as e:
        if verbose:
            print(f"\tEmployee.set_contacts() raised wrong exception when type is wrong: {e}")
        generates_type_err = False
    else:
        if verbose:
            print(f"\tEmployee.set_contacts() did not raise an exception when type is wrong")
        generates_type_err = False

    try:
        test_employee = Employee(driver=True, front_seater=True)
        emp2 = Employee(driver=True, front_seater=True)
        emp3 = Employee(driver=True, front_seater=True)
        test_employee.set_contacts([(emp2, 0.5), (emp3, 0.1)])
    except Exception as e:
        if verbose:
            print(f"\tEmployee.set_contacts() raised exception when type's are correct: {e}")
        generates_type_err = False
    else:
        pass

    if generates_type_err:
        points += 0.05

    for _ in range(15):
        test_employee = Employee(driver=random.choice([True, False]), front_seater=random.choice([True, False]))
        others = [
            (Employee(driver=random.choice([True, False]), front_seater=random.choice([True, False])), round(random.uniform(-1, 1), 2))
            for _ in range(random.randint(1, 3))
        ]

        try:
            test_employee.set_contacts(others)
        except Exception as e:
            print(f"\tEmployee.set_contacts() raised an unexpected exception: {e}")
        else:
            success = True
            for emp, val in others:
                try:
                    if test_employee.contacts[emp] != val:
                        if verbose:
                            print(f"\tEmployee.set_contacts() did not update dictionary correctly")
                        success = False
                except Exception as e:
                    if verbose:
                        print(f"\tEmployee.set_contacts() did not update dictionary correctly")
                        print(f"\tError when checking weights: {e}")
                    success = False

            try:
                if len(test_employee.contacts) != len(others):
                    if verbose:
                        print(f"\tEmployee.set_contacts() did not update dictionary correctly")
                    success = False
            except Exception as e:
                if verbose:
                    print(f"\tEmployee.set_contacts() did not update dictionary correctly")
                    print(f"\tError when checking number of keys: {e}")
                success = False

            if success:
                points += 0.01

            # generates ValueError?
            extra_emp = random.choice(others)
            try:
                test_employee.set_contacts([extra_emp])
            except ValueError as e:
                pass
            except Exception as e:
                if verbose:
                    print(f"\tEmployee.set_contacts() raised wrong exception when employee already present: {e}")
                generates_value_err = False
            else:
                if verbose:
                    print(f"\tEmployee.set_contacts() did not raise exception when employee already present")
                generates_value_err = False

    if generates_value_err:
        points += 0.05

    print("Testing Employee.set_contacts() finished: {:.2f}/0.25 points".format(points))
    return points


def test_get_car_weight(verbose=False):
    print("Testing Employee.get_car_weight():")
    points = 0

    with open("employee_weights.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for test_idx, (p_weights, _, p_drives, p_front, corr_w, _) in enumerate(test_cases):
        # build car for scenario
        p_contacts = dict()
        test_car = Car()
        test_car.seats = [None, None, None, None]
        for idx, (weight, is_contact) in enumerate(p_weights):
            if weight is not None:
                driver = True if idx == 0 else random.choice([True, False])
                front_sitter = True if driver else random.choice([True, False])
                new_pass = Employee(driver, front_sitter)
                test_car.seats[idx] = new_pass
                if is_contact:
                    p_contacts[new_pass] = weight

        # prepare passenger for scenario
        test_emp = Employee(p_drives, p_front)
        test_emp.contacts = p_contacts

        try:
            st_res = test_emp.get_car_weight(test_car)
        except Exception as e:
            if verbose:
                print(f"\tEmployee.get_car_weight() raised an error:")
                print(f"\t{e}")
        else:
            if not isinstance(st_res, (int, float)):
                if verbose:
                    print(f"\tEmployee.get_car_weight() returned wrong type; expected float, got {type(st_res)}")
            elif abs(corr_w - st_res) > 1e-4:
                if verbose:
                    print(f"\tEmployee.get_car_weight() returned wrong value (test case {test_idx + 1})")
                    print(f"\tExpected {corr_w}, got {st_res} for weights {p_weights}")
            else:
                correct += 1

    points = (correct / all_tests) * 0.5
    print("Testing Employee.get_car_weight() finished: {:.2f}/0.5 points".format(points))
    return points


def test_get_satisfaction(verbose=True):
    print("Testing Employee.get_satisfaction():")
    points = 0

    with open("employee_weights.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)
    handles_incorrect = True

    for test_idx, (p_weights, p_idx, p_drives, p_front, _, corr_sat) in enumerate(test_cases):
        # build car for scenario
        p_contacts = dict()
        test_car = Car()
        test_car.seats = [None, None, None, None]
        for idx, (weight, is_contact) in enumerate(p_weights):
            if weight is not None:
                driver = True if idx == 0 else random.choice([True, False])
                front_sitter = True if driver else random.choice([True, False])
                new_pass = Employee(driver, front_sitter)
                test_car.seats[idx] = new_pass
                if is_contact:
                    p_contacts[new_pass] = weight

        # prepare passenger for scenario
        test_emp = Employee(p_drives, p_front)
        test_emp.contacts = p_contacts
        # should return -1 if not in car
        try:
            st_res = test_emp.get_satisfaction(test_car)
            if st_res != -1:
                handles_incorrect = False
        except Exception as e:
            if verbose:
                print(f"\tEmployee.get_satisfaction() raised an error:")
                print(f"\t{e}")

        test_car.seats[p_idx] = test_emp

        try:
            st_res = test_emp.get_satisfaction(test_car)
        except Exception as e:
            if verbose:
                print(f"\tEmployee.get_satisfaction() raised an error:")
                print(f"\t{e}")
        else:
            if not isinstance(st_res, (int, float)):
                if verbose:
                    print(f"\tEmployee.get_satisfaction() returned wrong type; expected float, got {type(st_res)}")
            elif abs(corr_sat - st_res) > 1e-4:
                if verbose:
                    print(f"\tEmployee.get_satisfaction() returned wrong value (test case {test_idx + 1})")
                    print(f"\tExpected {corr_sat}, got {st_res} for weights {p_weights}")
            else:
                correct += 1

    points = (correct / all_tests) * 0.5
    if not handles_incorrect:
        points -= 0.1
    print("Testing Employee.get_satisfaction() finished: {:.2f}/0.5 points".format(points))
    return points


def test_choose_car(verbose=True):
    print("Testing Employee.choose_car():")
    points = 0

    with open("employee_cars.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for test_idx, (test_cars, p_drives, p_front, corr_car, _) in enumerate(test_cases):
        # build car for scenario
        p_contacts = dict()
        cars = list()
        for car_passengers in test_cars:
            test_car = Car()
            test_car.seats = [None, None, None, None]
            for idx, (weight, is_contact) in enumerate(car_passengers):
                if weight is not None:
                    driver = True if idx == 0 else random.choice([True, False])
                    front_sitter = True if driver else random.choice([True, False])
                    new_pass = Employee(driver, front_sitter)
                    test_car.seats[idx] = new_pass
                    if is_contact:
                        p_contacts[new_pass] = weight
            cars.append(test_car)

        # prepare passenger for scenario
        test_emp = Employee(p_drives, p_front)
        test_emp.contacts = p_contacts

        try:
            st_res = test_emp.choose_car(cars)
            st_idx = cars.index(st_res)
        except Exception as e:
            if verbose:
                print(f"\tEmployee.choose_car() raised an error:")
                print(f"\t{e}")
        else:
            if not isinstance(st_res, Car):
                if verbose:
                    print(f"\tEmployee.choose_car() returned wrong type; expected Car, got {type(st_res)}")
            elif corr_car != st_idx:
                if verbose:
                    print(f"\tEmployee.choose_car() returned wrong value (test case {test_idx + 1})")
                    print(f"\tExpected car with index {corr_car}, got {st_idx} for weights {test_cars}")
            else:
                correct += 1

    points = (correct / all_tests) * 0.5
    print("Testing Employee.choose_car() finished: {:.2f}/0.5 points".format(points))
    return points


def test_take_seat(verbose=False):
    print("Testing Employee.take_seat():")
    points = 0

    with open("employee_cars.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for test_idx, (test_cars, p_drives, p_front, _, corr_seat) in enumerate(test_cases):
        # build car for scenario
        p_contacts = dict()
        cars = list()
        for car_passengers in test_cars:
            test_car = Car()
            test_car.seats = [None, None, None, None]
            for idx, (weight, is_contact) in enumerate(car_passengers):
                if weight is not None:
                    driver = True if idx == 0 else random.choice([True, False])
                    front_sitter = True if driver else random.choice([True, False])
                    new_pass = Employee(driver, front_sitter)
                    test_car.seats[idx] = new_pass
                    if is_contact:
                        p_contacts[new_pass] = weight
            cars.append(test_car)

        # prepare passenger for scenario
        test_emp = Employee(p_drives, p_front)
        test_emp.contacts = p_contacts

        try:
            st_res = test_emp.take_seat(cars)
        except Exception as e:
            if verbose:
                print(f"\tEmployee.take_seat() raised an error:")
                print(f"\t{e}")
        else:
            if not isinstance(st_res, tuple):
                if verbose:
                    print(f"\tEmployee.take_seat() returned wrong type; expected tuple, got {type(st_res)}")
            elif len(st_res) != 2:
                if verbose:
                    print(f"\tEmployee.take_seat() returned wrong value; expected tuple with two values, found {len(st_res)}")
            elif not isinstance(st_res[0], int) or not isinstance(st_res[1], int):
                if verbose:
                    print(f"\tEmployee.take_seat() returned wrong value; expected tuples of integers, found ({type(st_res[0])}, {type(st_res[1])})")
            elif corr_seat != st_res:
                acceptable = False
                if st_res[0] == corr_seat[0] and ((st_res[1] == 2 and corr_seat[1] == 3) or (st_res[1] == 3 and corr_seat[1] == 2)):
                    acceptable = True
                if p_drives and corr_seat[1] == 0 and st_res[1] != 0:
                    acceptable = False
                if not p_drives and corr_seat[1] != 1:
                    acceptable = True
                if not acceptable and verbose:
                    print(f"\tEmployee.take_seat() returned wrong value (test case {test_idx + 1})")
                    print(f"\tExpected {corr_seat}, got {st_res} for weights {test_cars}")
                    print(f"\tEmployee drives: {p_drives}, sits_in_front: {p_front}")
                
                if acceptable:
                    correct += 1
            else:
                correct += 1

    points = correct / all_tests
    print("Testing Employee.take_seat() finished: {:.2f}/1 point".format(points))
    return points


def main():
    total = 0
    total += test_constructor()

    total += test_set_contacts()

    total += test_get_car_weight()

    total += test_get_satisfaction()

    total += test_choose_car()

    total += test_take_seat()

    print()
    print("EMPLOYEE: {:.2f}/3 points".format(total))


if __name__ == '__main__':
    main()
