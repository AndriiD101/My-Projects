import pickle
import random

from news import News
from person import Person
from population import Population, HomogeneousPopulation


def test_structure(test):
    try:
        if not isinstance(test.people, list):
            raise TypeError(
                "Population item has wrong type of people attribute. People should be list, is {}".format(type(test.people))
            )
    except AttributeError:
        raise AttributeError(
            "Population item missing attribute people. Do not alter class structure")

    try:
        if not isinstance(test.active_news, list):
            raise TypeError(
                "Population item has wrong type of active_news attribute. Active_news should be list, is {}".format(type(test.active_news))
            )
    except AttributeError:
        raise AttributeError(
            "Population item missing attribute active_news. Do not alter class structure")

    if isinstance(test, HomogeneousPopulation):
        try:
            if not isinstance(test.category, str):
                raise TypeError(
                    "HomogeneousPopulation item has wrong type of category attribute. Category should be string, is {}".format(type(test.category))
                )
        except AttributeError:
            raise AttributeError(
                "HomogeneousPopulation item missing attribute category. Do not alter class structure")


def test_generate_population():
    print("Testing Population.generate_population()...")

    for _ in range(10):
        size = random.choice([50, 100, 250, 500, 1000])
        friend_count = random.randint(1, 10)
        patience_interval = random.choice([(3, 8), (4, 6), (3, 10)])

        try:
            test = Population(size, friend_count, patience_interval)
        except Exception:
            print("\tPopulation.__init__() failed. Could not create Population object")
            return

        test_structure(test)

        if len(test.people) != size:
            print("\tIncorrect population initialization: people list has {} elements instead of {}".format(len(test.people), size))
            return

        for elem in test.people:
            if not isinstance(elem, Person):
                print("\tIncorrect population element type: all members of the population should be Person objects. Got {}".format(type(elem)))
                return

            try:
                if len(elem.friends_list) != friend_count:
                    print("\tNot all population members have {} friends. Got {}".format(friend_count, len(elem.friends_list)))
                    return
            except AttributeError:
                print("\tCould not find attribute friends_list for population element")
                return
            except TypeError:
                print("\tFriends_list attribute for population element is not list (got {})".format(type(elem.friends_list)))
                return

            unique_friends = set(elem.friends_list)
            if len(unique_friends) != friend_count:
                print("\tPopulation member doesn't have unique friends only. Got {} unique friends instead of {}".format(len(unique_friends), friend_count))
                return

            if elem in unique_friends:
                print("\tA population member cannot be his/her own friend")
                return

    print("Great news! Population.generate_population() passed all tests!")

def test_introduce_news():
    print("Testing Population.introduce_news()...")

    with open('introduce_news_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for active, to_add in test_list:
        size = random.choice([50, 100, 250, 500, 1000])
        friend_count = random.randint(1, 10)
        patience_interval = random.choice([(3, 8), (4, 6), (3, 10)])

        try:
            test_pop = Population(size, friend_count, patience_interval)
        except Exception:
            print("\tCould not create Population object")
            return

        test_structure(test_pop)

        orig_length = len(active)
        test_pop.active_news = active

        try:
            res = test_pop.introduce_news(to_add)
        except Exception:
            print("\tCould not execute Population.introduce_news()")
            return

        test_structure(test_pop)

        new_length = len(test_pop.active_news)
        if new_length != orig_length + 1:
            print("\tDid not add news item correctly to the list of active news items")
            print("\tExpected {} active news, got {}".format(orig_length + 1, new_length))
            return

        if to_add not in test_pop.active_news:
            print("Did not add news item to the list of active news items")
            return

        if not isinstance(res, list):
            print("\tIncorrect return type of Population.introduce_news()")
            print("\tExpected list, got {}".format(type(res)))
            return

        if len(res) != 5:
            print("\tIncorrect return value of Population.introduce_news()")
            print("\tExpected list of length 5, got {}".format(len(res)))

        unique_people = set(res)
        if len(unique_people) != 5:
            print("\tIncorrect return value of Population.introduce_news()")
            print("\tExpected list of 5 unique persons, got {}".format(len(unique_people)))

        for elem in res:
            if not isinstance(elem, Person):
                print("\tIncorrect return value of Population.introduce_news()")
                print("\tReturn list contains elements of type {}".format(type(elem)))
                return

            if to_add.category not in elem.interested_in:
                print("\tIncorrect return value of Population.introduce_news()")
                print("\tCannot send news of {} category to person interested in {}".format(to_add.category, elem.interested_in))

    print("Great news! Population.introduce_news() passed all tests!")


def test_update_news():
    print("Testing Population.update_news()...")

    with open('update_news_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for orig, step, correct in test_list:
        size = random.choice([50, 100, 250, 500, 1000])
        friend_count = random.randint(1, 10)
        patience_interval = random.choice([(3, 8), (4, 6), (3, 10)])

        try:
            test_pop = Population(size, friend_count, patience_interval)
        except Exception:
            print("\tCould not create Population object")
            return

        test_structure(test_pop)

        corr_length = len(correct)

        test_pop.active_news = orig
        try:
            res = test_pop.update_news(step)
        except Exception as e:
            print(f"\tCould not execute Population.update_news(): {e}")
            return

        if res is not None:
            print("\tIncorrect return value of Population.update_news()")
            print("\tExpected None, got {}".format(res))
            return

        test_structure(test_pop)

        new_length = len(test_pop.active_news)
        if new_length != corr_length:
            print("\tDid not update list of active news items correctly")
            print("\tExpected {} active news, got {}".format(corr_length, new_length))
            return

        for elem in test_pop.active_news:
            if not isinstance(elem, News):
                print("\tList of active news has element of type {}".format(type(elem)))
                return

    print("Great news! Population.update_news() passed all tests!")


def test_count_readers():
    print("Testing Population.count_readers()...")

    with open('count_readers_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for people, news, corr in test_list:
        try:
            test_pop = Population(100, 6, (3, 8))
        except Exception:
            print("\tCould not create Population object")
            return

        test_structure(test_pop)
        test_pop.people = people

        try:
            res = test_pop.count_readers(news)
        except Exception:
            print("\tCould not execute Population.count_readers()")
            return

        if not isinstance(res, int):
            print("\tIncorrect return type of Population.count_readers()")
            print("\tExpected int, got {}".format(type(res)))
            return

        if res != corr:
            print("\tIncorrect return value of Population.count_readers()")
            print("\tExpected {}, got {}".format(corr, res))
            return

    print("Great news! Population.count_readers() passed all tests!")


def test_get_number_of_interested():
    print("Testing Population.get_number_of_interested()...")

    with open('get_number_of_interested_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for people, cat, corr in test_list:
        try:
            test_pop = Population(100, 6, (3, 8))
        except Exception:
            print("\tCould not create Population object")
            return

        test_structure(test_pop)
        test_pop.people = people

        try:
            res = test_pop.get_number_of_interested(cat)
        except Exception:
            print("\tCould not execute Population.get_number_of_interested()")
            return

        if not isinstance(res, int):
            print("\tIncorrect return type of Population.get_number_of_interested()")
            print("\tExpected int, got {}".format(type(res)))
            return

        if res != corr:
            print("\tIncorrect return value of Population.get_number_of_interested()")
            print("\tExpected {}, got {}".format(corr, res))
            return

    print("Great news! Population.get_number_of_interested() passed all tests!")

def test_generate_population_homogeneous():
    print("Testing HomogeneousPopulation.generate_population()...")

    for cat in ["politics", "world", "culture", "tech", "local", "sport"]:
        size = random.choice([50, 100, 250, 500, 1000])
        friend_count = random.randint(1, 10)
        patience_interval = random.choice([(3, 8), (4, 6), (3, 10)])

        try:
            test = HomogeneousPopulation(size, friend_count, patience_interval, cat)
        except Exception as e:
            print(f"\tHomogeneousPopulation.__init__() failed. Could not create Population object: {e}")
            return

        test_structure(test)

        if len(test.people) != size:
            print("\tIncorrect population initialization: people list has {} elements instead of {}".format(len(test.people), size))
            return

        for elem in test.people:
            if not isinstance(elem, Person):
                print("\tIncorrect population element type: all members of the population should be Person objects. Got {}".format(type(elem)))
                return

            try:
                if len(elem.friends_list) != friend_count:
                    print("\tNot all population members have {} friends. Got {}".format(friend_count, len(elem.friends_list)))
                    return
            except AttributeError:
                print("\tCould not find attribute friends_list for population element")
                return
            except TypeError:
                print("\tFriends_list attribute for population element is not list (got {})".format(type(elem.friends_list)))
                return

            unique_friends = set(elem.friends_list)
            if len(unique_friends) != friend_count:
                print("\tPopulation member doesn't have unique friends only. Got {} unique friends instead of {}".format(len(unique_friends), friend_count))
                return

            if elem in unique_friends:
                print("\tA population member cannot be his/her own friend")
                return

            try:
                if cat not in elem.interested_in:
                    print("\tNot all population members have interest {}. Got {}".format(cat, elem.interested_in))
                    return
            except AttributeError:
                print("\tCould not find attribute interested_in for population element")
                return
            except TypeError:
                print("\tInterested_in attribute for population element is not list (got {})".format(type(elem.interested_in)))
                return

    print("Great news! HomogeneousPopulation.generate_population() passed all tests!")


def main():
    test_generate_population()
    print()

    test_introduce_news()
    print()

    test_update_news()
    print()

    test_count_readers()
    print()

    test_get_number_of_interested()
    print()

    test_generate_population_homogeneous()
    print()


if __name__ == '__main__':
    main()
