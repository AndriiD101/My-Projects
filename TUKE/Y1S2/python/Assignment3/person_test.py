import pickle
import random

from person import Person


def test_structure(test):
    try:
        if not isinstance(test.threshold, float):
            raise TypeError(
                "Person item has wrong type of threshold attribute. Threshold should be float, is {}".format(type(test.threshold))
            )
    except AttributeError:
        raise AttributeError(
            "Person item missing attribute threshold. Do not alter class structure")

    try:
        if not isinstance(test.interested_in, list):
            raise TypeError(
                "Person item has wrong type of interested_in attribute. Interested_in should be list, is {}".format(type(test.interested_in))
            )
    except AttributeError:
        raise AttributeError(
            "Person item missing attribute interested_in. Do not alter class structure")

    try:
        if not isinstance(test.friends_list, list):
            raise TypeError(
                "Person item has wrong type of friends_list attribute. Friends_list should be list, is {}".format(type(test.friends_list))
            )
    except AttributeError:
        raise AttributeError(
            "Person item missing attribute friends_list. Do not alter class structure")

    try:
        if not isinstance(test.has_read, list):
            raise TypeError(
                "Person item has wrong type of has_read attribute. has_read should be list, is {}".format(type(test.has_read))
            )
    except AttributeError:
        raise AttributeError(
            "Person item missing attribute has_read. Do not alter class structure")


def test_is_interested_in():
    print("Testing Person.is_interested_in()...")

    with open('is_interested_in_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for threshold, interests, patience, cat, res in test_list:
        try:
            person = Person(threshold, interests, patience)
        except Exception:
            print("\tPerson.__init__() failed. Do not alter constructor!")
            return

        test_structure(person)

        try:
            stud_res = person.is_interested_in(cat)
        except Exception:
            print("\tPerson.is_interested_in() generated error for category {}".format(cat))
            print("\tPerson attributes: {}, {}".format(threshold, interests))
            return

        if not isinstance(stud_res, bool):
            print("\tWrong return type for Person.is_interested_in(). Expected bool, got {}".format(type(stud_res)))
            return

        if stud_res != res:
            print("\tPerson.is_interested_in() returned wrong value: expected {}, got {}".format(res, stud_res))
            print("\tPerson interests: {}; looking for category {}".format(interests, cat))
            return

    print("Great news! Person.is_interested_in() passed all tests!")


def test_has_read_news():
    print("Testing Person.has_read_news()...")

    with open('has_read_news_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for threshold, interests, patience, news, news_piece, res in test_list:
        try:
            person = Person(threshold, interests, patience)
        except Exception:
            print("\tPerson.__init__() failed. Do not alter constructor!")
            return

        test_structure(person)
        person.has_read = news

        try:
            stud_res = person.has_read_news(news_piece)
        except Exception:
            print("\tPerson.has_read_news() generated error")
            return

        if not isinstance(stud_res, bool):
            print("\tWrong return type for Person.has_read_news(). Expected bool, got {}".format(type(stud_res)))
            return

        if stud_res != res:
            print("\tPerson.has_read_news() returned wrong value: expected {}, got {}".format(res, stud_res))
            print("\tPerson has_read: {}; looking for news {}".format(news, news_piece))
            return

    print("Great news! Person.has_read_news() passed all tests!")


def test_make_friends():
    print("Testing Person.make_friends()...")

    with open('make_friends_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for population, count in test_list:
        try:
            person = Person(random.random(), ["world", "tech", "sport"], random.randint(3, 8))
        except Exception:
            print("\tPerson.__init__() failed. Do not alter constructor!")
            return

        test_structure(person)

        try:
            stud_res = person.make_friends(population, count)
        except Exception as e:
            print("\tPerson.make_friends() generated error")
            return

        if stud_res is not None:
            print("\tPerson.make_friends() should not have a return value. Expected None, got {}".format(stud_res))
            return

        try:
            if len(person.friends_list) != count:
                print("\tPerson.make_friends() didn't update friends_list correctly.")
                print("\tExpected {} friends, got {}".format(count, len(person.friends_list)))
                return

            unique_friends = set(person.friends_list)
            if len(unique_friends) != count:
                print("\tPerson.make_friends() didn't select unique friends.")
                print("\tExpected {} unique friends, got {}".format(count, len(unique_friends)))
                return

            for friend in unique_friends:
                if friend == person:
                    print("\tIncorrect implementation in Person.make_friends() - person cannot be his/her own friend.")
                    return
                if friend not in population:
                    print("\tPerson.make_friends() didn't select friends only from population")
                    print("\tFriend object {} not found in population".format(friend))
                    return
        except AttributeError:
            print("\tCould not find attribute Person.friends_list after calling Person.make_friends()")
            return
        except Exception:
            print("\tCould not check Person.make_friends()")
            return

    print("Great news! Person.make_friends() passed all tests!")


def test_process_news():
    print("Testing Person.process_news()...")

    with open('process_news_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)
        # print(test_list)

    for person, news_read, test_news, step, res in test_list:
        # print(res)
        try:
            person.has_read = news_read
            stud_res = person.process_news(test_news, step)
        except Exception as e:
            print(f"\tPerson.process_news() generated error: {e}")
            return

        if not isinstance(stud_res, list):
            print("\tPerson.process_news() has wrong return type. Expected list, got {}".format(type(stud_res)))
            return

        if len(stud_res) != len(res):
            # print(res)
            print("\tPerson.process_news() does not send news to all friends with interest")
            print("\tExpected {} friends, got {}".format(len(res), len(stud_res)))
            return

    print("Great news! Person.process_news() passed all tests!")


def main():
    test_is_interested_in()
    print()

    test_has_read_news()
    print()

    test_make_friends()
    print()

    test_process_news()
    print()


if __name__ == '__main__':
    main()
