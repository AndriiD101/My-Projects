import pickle

import assignment1 as a1


def test_throw_to_points(verbose=False):
    print("Testing throw_to_points():")

    with open("points.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (test_input, result) in enumerate(test_cases, start=1):
        try:
            st_res = a1.throw_to_points(test_input)
        except Exception as e:
            if verbose:
                print("\tAn exception occurred when calling throw_to_points() (test case {}):".format(idx))
                print("\t", e)
        else:
            # must be list of tuples, each tuple must have 2 tuples at most, inner tuples 3 values - string or none
            if not isinstance(st_res, int):
                if verbose:
                    print("\tthrow_to_points() returned wrong type. Expected int, got {}".format(type(st_res)))
                continue

            if st_res != result:
                if verbose:
                    print("\tthrow_to_points() returned wrong value. Expected {}, got {} (for input {})".format(result, st_res, test_input))
            else:
                correct += 1

    points = (correct / all_tests) * 0.5
    print("Testing throw_to_points() finished: {:.2f}/0.5 points".format(points))
    return points


def test_parse_throws(verbose=False):
    print("Testing parse_throws():")

    with open("parsing.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (test_input, result) in enumerate(test_cases):
        try:
            st_res = a1.parse_throws(test_input)
        except Exception as e:
            if verbose:
                print("\tAn exception occurred when calling parse_throws() (test case {}):".format(idx + 1))
                print("\t", e)
        else:
            # must be list of tuples, each tuple must have 2 tuples at most, inner tuples 3 values - string or none
            if not isinstance(st_res, list):
                if verbose:
                    print("\tparse_throws() returned wrong type. Expected list, got {}".format(type(st_res)))
                continue

            for elem in st_res:
                if not isinstance(elem, tuple):
                    if verbose:
                        print("\tparse_throws() returned wrong type. Expected list of tuples, found {} in list".format(type(elem)))
                    continue

                if len(elem) != 1 and len(elem) != 2:
                    if verbose:
                        print("\tparse_throws() returned wrong value. Expected tuples for two players, found tuple with {} elements ({})".format(len(elem), elem))
                    continue

                for throw_group in elem:
                    if not isinstance(throw_group, tuple):
                        if verbose:
                            print("\tparse_throws() returned wrong value. Data for two players should be a tuple of tuples, found {} in {}".format(type(throw_group), elem))
                        continue

                    if len(throw_group) == 0 or len(throw_group) > 3:
                        if verbose:
                            print("\tparse_throws() returned wrong value. One player can throw three darts at most and one at least, found {} in {}".format(len(throw_group), throw_group))
                        continue

            correct += st_res == result

    points = (correct / all_tests) * 2
    print("Testing parse_throws() finished: {:.2f}/2 points".format(points))
    return points


def test_save_throws(verbose=False):
    print("Testing save_throws():")

    with open("saving.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, test_input in enumerate(test_cases, start=1):
        try:
            a1.save_throws(test_input, "save_file.txt")
        except Exception as e:
            if verbose:
                print("\tAn exception occurred when calling save_throws() (test case {}):".format(idx))
                print("\t", e)
        else:
            with open("sample_files\\save_file{}.txt".format(idx)) as corr_file:
                correct_lines = corr_file.readlines()

            try:
                with open("save_file.txt") as st_file:
                    st_lines = st_file.readlines()
            except FileNotFoundError:
                continue

            if len(st_lines) != len(correct_lines):
                if verbose:
                    print("\tIncorrect file generated (test case {}). Expected {} lines, found {}".format(idx, len(correct_lines), len(st_lines)))
            else:
                for line_id, (st_line, corr_line) in enumerate(zip(st_lines, correct_lines), start=1):
                    if st_line != corr_line:
                        if verbose:
                            print("\tIncorrect value found in line {} (test case {}). Expected {}, found {}".format(line_id, idx, corr_line, st_line))
                        break
                else:
                    correct += 1

    points = (correct / all_tests)
    print("Testing save_throws() finished: {:.2f}/1 point".format(points))
    return points


def test_split_into_sets(verbose=False):
    print("Testing split_into_sets():")

    with open("split.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (test_input, result) in enumerate(test_cases, start=1):
        print("--------------------------------")
        print(test_input)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(result)
        try:
            st_res = a1.split_into_sets(test_input)
        except Exception as e:
            if verbose:
                print("\tAn exception occurred when calling split_into_sets() (test case {}):".format(idx))
                print("\t", e)
        else:
            # must be list of tuples, each tuple must have 2 tuples at most, inner tuples 3 values - string or none
            if not isinstance(st_res, list):
                if verbose:
                    print("\tsplit_into_sets() returned wrong type. Expected list, got {}".format(type(st_res)))
                continue

            if len(st_res) != len(result):
                if verbose:
                    print("\tIncorrect number of sets. Expected {}, got {} (test case {})".format(len(result), len(st_res), idx))
                continue

            for set_idx, (st_set, corr_set) in enumerate(zip(st_res, result), start=1):
                if not isinstance(st_set, list):
                    if verbose:
                        print("\tsplit_into_sets() returned wrong type. Expected list of lists, found {} in list".format(type(st_set)))
                    continue

                if len(st_set) != len(corr_set):
                    if verbose:
                        print("\tIncorrect number of legs in set {}. Expected {}, got {} (test case {})".format(set_idx, len(corr_set), len(st_set), idx))
                    continue

                for leg_idx, (st_leg, corr_leg) in enumerate(zip(st_set, corr_set), start=1):
                    if not isinstance(st_leg, list):
                        if verbose:
                            print("\tsplit_into_sets() returned wrong type. Expected list of lists for set, found {} in list".format(type(st_leg)))
                        continue

                    if len(st_leg) != len(corr_leg):
                        if verbose:
                            print("\tIncorrect number of throws in leg {} in set {}. Expected {}, got {} (test case {})".format(leg_idx, set_idx, len(corr_leg), len(st_leg), idx))
                        continue

                    for elem in st_leg:
                        if not isinstance(elem, tuple):
                            if verbose:
                                print("\tsplit_into_sets() returned wrong type. Expected list of tuples for a leg, found {} in list".format(type(elem)))
                            continue

                        if len(elem) != 1 and len(elem) != 2:
                            if verbose:
                                print("\tsplit_into_sets() returned wrong value. Expected tuples for two players, found tuple with {} elements ({})".format(len(elem), elem))
                            continue

                        for throw_group in elem:
                            if not isinstance(throw_group, tuple):
                                if verbose:
                                    print("\tsplit_into_sets() returned wrong value. Data for two players should be a tuple of tuples, found {} in {}".format(type(throw_group), elem))
                                continue

                            if len(throw_group) == 0 or len(throw_group) > 3:
                                if verbose:
                                    print("\tsplit_into_sets() returned wrong value. One player can throw three darts at most and one at least, found {} in {}".format(len(throw_group), throw_group))
                                continue

            correct += st_res == result

    points = (correct / all_tests) * 2
    print("Testing split_into_sets() finished: {:.2f}/2 points".format(points))
    return points


def test_get_match_result(verbose=False):
    print("Testing get_match_result():")

    with open("game_results.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases) * 2

    for idx, (test_input, result) in enumerate(test_cases, start=1):
        # print("--------------------------------")
        # print(test_input)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(result)
        try:
            st_res = a1.get_match_result(test_input)
        except Exception as e:
            if verbose:
                print("\tAn exception occurred when calling get_match_result() (test case {}):".format(idx))
                print("\t", e)
        else:
            # must be list of tuples, each tuple must have 2 tuples at most, inner tuples 3 values - string or none
            if not isinstance(st_res, tuple):
                if verbose:
                    print("\tget_match_result() returned wrong type. Expected tuple with two values, got {}".format(type(st_res)))
                continue

            if len(st_res) != 2:
                if verbose:
                    print("\tget_match_result() returned wrong type. Expected tuple with two values, got {} (test case {})".format(len(st_res), idx))
                continue

            game_result, set_results = st_res
            if not isinstance(game_result, tuple):
                if verbose:
                    print("\tget_match_result() returned wrong type. Game result should be represented as a tuple, got {}".format(type(game_result)))
                continue

            if len(game_result) != 2:
                if verbose:
                    print("\tget_match_result() returned wrong value. Game result should be a tuple of two numbers, got {}".format(len(game_result)))
                continue

            if not isinstance(set_results, list):
                if verbose:
                    print("\tget_match_result() returned wrong type. Set results should be represented as a list, got {}".format(type(set_results)))
                continue

            for set_res in set_results:
                if not isinstance(set_res, tuple):
                    if verbose:
                        print("\tget_match_result() returned wrong type. Set results should be a list of tuples, got {}".format(type(set_res)))
                    break

                if len(set_res) != 2:
                    if verbose:
                        print("\tget_match_result() returned wrong value. Set results should be tuples of two values, got {}".format(len(set_res)))
                    break
            else:
                correct += game_result == result[0]
                correct += set_results == result[1]

    points = (correct / all_tests)
    print("Testing get_match_result() finished: {:.2f}/1 point".format(points))
    return points


def test_get_180s(verbose=False):
    print("Testing get_180s():")

    with open("180s.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (test_input, result) in enumerate(test_cases, start=1):
        try:
            st_res = a1.get_180s(test_input)
        except Exception as e:
            if verbose:
                print("\tAn exception occurred when calling get_180s() (test case {}):".format(idx))
                print("\t", e)
        else:
            # must be list of tuples, each tuple must have 2 tuples at most, inner tuples 3 values - string or none
            if not isinstance(st_res, tuple):
                if verbose:
                    print("\tget_180s() returned wrong type. Expected tuple with two values, got {}".format(type(st_res)))
                continue

            if len(st_res) != 2:
                if verbose:
                    print("\tget_180s() returned wrong type. Expected tuple with two values, got {} (test case {})".format(len(st_res), idx))
                continue

            correct += st_res == result

    points = (correct / all_tests) * 0.5
    print("Testing get_180s() finished: {:.2f}/0.5 points".format(points))
    return points


def test_get_average_throws(verbose=False):
    print("Testing get_average_throws():")

    with open("averages.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases)

    for idx, (test_input, result) in enumerate(test_cases, start=1):
        # print("-----------------------------------")
        # print(test_input)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(result)
        try:
            st_res = a1.get_average_throws(test_input)
        except Exception as e:
            if verbose:
                print("\tAn exception occurred when calling get_average_throws() (test case {}):".format(idx))
                print("\t", e)
        else:
            # must be list of tuples, each tuple must have 2 tuples at most, inner tuples 3 values - string or none
            if not isinstance(st_res, list):
                if verbose:
                    print("\tget_average_throws() returned wrong type. Expected list, got {}".format(type(st_res)))
                continue

            if len(st_res) != len(result):
                if verbose:
                    print("\tget_average_throws() returned wrong value. Expected list with {} elements, got {} (test case {})".format(len(result), len(st_res), idx))
                continue

            for elem in st_res:
                if not isinstance(elem, tuple):
                    if verbose:
                        print("\tget_average_throws() returned wrong type. Expected list of tuples, found {}".format(type(elem)))
                    break
                if len(elem) != 2:
                    if verbose:
                        print("\tget_average_throws() returned wrong type. Expected list of tuples with two values, found {}".format(len(elem)))
                    break
            else:
                for st_set_avg, corr_set_avg in zip(st_res, result):
                    p1_avg, p2_avg = st_set_avg
                    p1_corr, p2_corr = corr_set_avg

                    if abs(p1_avg - p1_corr) > 0.1:
                        break
                    if abs(p2_avg - p2_corr) > 0.1:
                        break
                else:
                    correct += st_res == result

    points = (correct / all_tests)
    print("Testing get_average_throws() finished: {:.2f}/1 point".format(points))
    return points


def test_generate_finishes(verbose=False):
    print("Testing generate_finishes():")

    with open("finishes.pkl", "rb") as test_file:
        test_cases = pickle.load(test_file)
    correct = 0
    all_tests = len(test_cases) * 2

    has_val = False
    for idx, (test_input, result) in enumerate(test_cases, start=1):
        # print("--------------------------------")
        # print(test_input)
        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # print(result)
        try:
            st_res = a1.generate_finishes(test_input)
        except Exception as e:
            if verbose:
                print("\tAn exception occurred when calling generate_finishes() (test case {}):".format(idx))
                print("\t", e)
        else:
            # must be list of tuples, each tuple must have 2 tuples at most, inner tuples 3 values - string or none
            if not isinstance(st_res, list):
                if verbose:
                    print("\tgenerate_finishes() returned wrong type. Expected list, got {}".format(type(st_res)))
                continue

            for elem in st_res:
                if not isinstance(elem, list):
                    if verbose:
                        print("\tgenerate_finishes() returned wrong type. Expected list of lists, found {}".format(type(elem)))
                    break
                if len(elem) > 3:
                    if verbose:
                        print("\tgenerate_finishes() returned wrong value. Expected list of lists with three values at most, found {}: {}".format(len(elem), elem))
                    break

                for throw in elem:
                    if not isinstance(throw, str):
                        if verbose:
                            print("\tgenerate_finishes() returned wrong type. Throws should be represented as strings, found {} in {}".format(throw, elem))
                        break

            # found all combinations?
            if len(st_res) != 0:
                has_val = True
            for combo in result:
                combo2 = combo.copy()
                if len(combo) == 3:
                    t1, t2, t3 = combo
                    combo2 = [t2, t1, t3]
                if combo not in st_res and combo2 not in st_res:
                    if verbose:
                        print("\tgenerate_finishes() returned wrong value. Possible finish {} not found in {}".foramt(combo, st_res))
                    break
            else:
                correct += 1

            # no duplicates?
            for combo in st_res:
                if len(combo) == 3:
                    t1, t2, t3 = combo
                    if t1 != t2 and [t2, t1, t3] in st_res:
                        if verbose:
                            print("\tDuplicate found in generate_finishes() result: {} and {} in {}".foramt(combo, [t2, t1, t3], st_res))
                        break
            else:
                correct += 1

    if not has_val:
        correct = False

    points = (correct / all_tests) * 2
    print("Testing generate_finishes() finished: {:.2f}/2 points".format(points))
    return points


def main():
    total = 0
    # total += test_throw_to_points()

    # total += test_parse_throws()

    # total += test_save_throws()

    total += test_split_into_sets()

    # total += test_get_match_result()

    # total += test_get_180s()

    # total += test_get_average_throws()

    # total += test_generate_finishes()

    print()
    print("TOTAL: {:.2f}/10 points".format(total))


if __name__ == "__main__":
    main()
