import pickle

from news import News
from simulation import *


def test_simulate_spread():
    print("Testing simulation.simulate_spread()")

    with open('simulate_spread_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for all_news, population in test_list:
        try:
            res = simulate_spread(all_news, population)
        except Exception:
            print("\tCould not execute simulation.simulate_spread() correctly")
            return

        if not isinstance(res, dict):
            print("\tIncorrect return type of simulation.simulate_spread()")
            print("\tExpected dict, got {}".format(type(res)))
            return

        if len(res) != len(all_news):
            print("\tIncorrect return value of simulation.simulate_spread()")
            print("\tExpected results for {} news items, got {}".format(len(all_news), len(res)))
            return

        for key in res:
            if not isinstance(key, News):
                print("\tIncorrect keys in simulation.simulate_spread() return value")
                print("\tAll keys should be of type News, got {}".format(type(key)))
                return

            if not isinstance(res[key], list):
                print("\tIncorrect values in simulation.simulate_spread() return value")
                print("\tAll values should be of type list, got {}".format(type(res[key])))
                return

            for idx, elem in enumerate(res[key]):
                if not isinstance(elem, int):
                    print("\tIncorrect values in simulation.simulate_spread() return value")
                    print("\tAll values should be lists of integers, got {}".format(type(elem)))
                    return

                if idx != 0:
                    if res[key][idx] < res[key][idx - 1]:
                        print("\tIncorrect values in simulation.simulate_spread() return value")
                        print("\tCount of readers cannot be a decreasing list {}".format(res[key]))
                        return

            if res[key][-1] != population.count_readers(key):
                print("\tIncorrect values in simulation.simulate_spread() return value")
                print("\tNews item has {} readers, but {} found in list".format(population.count_readers(key), res[key][-1]))
                return

        lengths = [len(res[key]) for key in res]
        if len(set(lengths)) != 1:
            print("\tIncorrect values in simulation.simulate_spread() return value")
            print("\tLists for each news item should have the same length, got list lengths {}".format(lengths))
            return

    print("Great news! simulation.simulate_spread() passed all tests!")


def test_average_spread_with_excitement_rate():
    print("Testing simulation.average_spread_with_excitement_rate()")

    with open('avg_spread_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)
        # print(test_list)

    for er, size, friends, patience_interval, tests, safe_avg in test_list:
        try:
            res = average_spread_with_excitement_rate(er, size, friends, patience_interval, tests)
        except Exception:
            print("\tCould not execute simulation.average_spread_with_excitement_rate() correctly")
            return

        if not isinstance(res, tuple):
            print("\tIncorrect return type of simulation.average_spread_with_excitement_rate()")
            print("\tExpected two values, got {}".format(type(res)))
            return

        if len(res) != 2:
            print("\tIncorrect number of return values of simulation.average_spread_with_excitement_rate()")
            print("\tExpected two values, got {}".format(len(res)))
            return

        reaches, avg = res

        if not isinstance(reaches, list):
            print("\tIncorrect type of first return value of simulation.average_spread_with_excitement_rate()")
            print("\tExpected list of reach counts, got {}".format(type(reaches)))
            return

        if len(reaches) != tests:
            print("\tIncorrect first return value of simulation.average_spread_with_excitement_rate()")
            print("\tExpected list with {} elements, got {}".format(tests, len(reaches)))
            return

        for elem in reaches:
            if not isinstance(elem, int):
                print("\tIncorrect type of first return value of simulation.average_spread_with_excitement_rate()")
                print("\tExpected list of integers, got list containing {}".format(type(elem)))
                return

        if not isinstance(avg, float):
            print("\tIncorrect type of second return value of simulation.average_spread_with_excitement_rate()")
            print("\tExpected average reach as float, got {}".format(type(avg)))
            return

        corr_avg = sum(reaches) / len(reaches)
        if abs(corr_avg - avg) > 0.001:
            print("\tIncorrectly calculated average in simulation.average_spread_with_excitement_rate()")
            print("\tExpected {}, got {} for list {}".format(corr_avg, avg, reaches))
            return

        if abs(safe_avg - avg) > 3:
            print("\tYou might have an incorrectly implemented simulation in simulation.average_spread_with_excitement_rate()")
            print("\tAverage spread with excitement rate {}, population size {}, {} friends per person is expected to be around {}, yours was {}".format(
                er, size, friends, safe_avg, avg
            ))

    print("Great news! simulation.average_spread_with_excitement_rate() passed all tests!")


def test_excitement_to_reach_percentage():
    print("Testing simulation.excitement_to_reach_percentage()")

    with open('er_to_reach_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)
        # print(test_list)

    for perc, size, friend_count, patience_interval, safe_avg in test_list:
        try:
            res = excitement_to_reach_percentage(perc, size, friend_count, patience_interval)
        except Exception:
            print("\tCould not execute simulation.excitement_to_reach_percentage() correctly")
            return

        if not isinstance(res, float):
            print("\tIncorrect return type of simulation.excitement_to_reach_percentage()")
            print("\tExpected float, got {}".format(type(res)))
            return

        if abs(res - safe_avg) > 0.05:
            print("\tIncorrect return value for simulation.excitement_to_reach_percentage()")
            print("\tExpected to get excitement rate around {}, got {}".format(safe_avg, res))
            print("\tParameters: percentage - {}, pop_size - {}, friends_count - {}".format(perc, size, friend_count))
            return

    print("Great news! simulation.excitement_to_reach_percentage() passed all tests!")


def test_excitement_to_reach_percentage_special_interest():
    print("Testing simulation.excitement_to_reach_percentage_special_interest()")

    with open('er_to_reach_special_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for perc, size, friend_count, patience_interval, cat, safe_avg in test_list:
        try:
            res = excitement_to_reach_percentage_special_interest(perc, size, friend_count, patience_interval, cat)
        except Exception:
            print("\tCould not execute simulation.excitement_to_reach_percentage_special_interest() correctly")
            return

        if not isinstance(res, float):
            print("\tIncorrect return type of simulation.excitement_to_reach_percentage_special_interest()")
            print("\tExpected float, got {}".format(type(res)))
            return

        if abs(res - safe_avg) > 0.05:
            print("\tIncorrect return value for simulation.excitement_to_reach_percentage_special_interest()")
            print("\tExpected to get excitement rate around {}, got {}".format(safe_avg, res))
            print("\tParameters: percentage - {}, pop_size - {}, friends_count - {}, category - {}".format(perc, size, friend_count, cat))
            return

    print("Great news! simulation.excitement_to_reach_percentage_special_interest() passed all tests!")


def main():
    test_simulate_spread()
    print()

    test_average_spread_with_excitement_rate()
    print()

    test_excitement_to_reach_percentage()
    print()

    test_excitement_to_reach_percentage_special_interest()
    print()


if __name__ == '__main__':
    main()
