import pickle

from news import News


def test_structure(test):
    try:
        if not isinstance(test.category, str):
            raise TypeError(
                "News item has wrong type of category attribute. Category should be string, is {}".format(type(test.category))
            )
    except AttributeError:
        raise AttributeError(
            "News item missing attribute category. Do not alter class structure")

    try:
        if not isinstance(test.excitement_rate, float):
            raise TypeError(
                "News item has wrong type of excitement_rate attribute. Excitement_rate should be float, is {}".format(type(test.excitement_rate))
            )
    except AttributeError:
        raise AttributeError(
            "News item missing attribute excitement_rate. Do not alter class structure")

    try:
        if not isinstance(test.validity_length, int):
            raise TypeError(
                "News item has wrong type of validity_length attribute. validity_length should be int, is {}".format(type(test.validity_length))
            )
    except AttributeError:
        raise AttributeError(
            "News item missing attribute validity_length. Do not alter class structure")

    try:
        if not isinstance(test.created, int):
            raise TypeError(
                "News item has wrong type of created attribute. created should be int, is {}".format(type(test.created))
            )
    except AttributeError:
        raise AttributeError(
            "News item missing attribute created. Do not alter class structure")


def test_check_data():
    print("Testing News.check_data()...")

    for cat in ["world", "culture", "tech"]:
        for er in [0.0, 0.3, 0.5, 1.0]:
            for valid in [1, 5, 10]:
                for created in [1, 5, 10]:
                    try:
                        test = News(cat, er, valid, created)
                        test_structure(test)
                    except Exception:
                        print("\tobject creation should not generate error for valid input")
                        print("\ttested on {}, {}, {}, {}".format(cat, er, valid, created))
                        return

    try:
        test = News("gossip", 0.5, 5, 1)
    except ValueError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of ValueError - invalid category".format(type(e)))
        return
    else:
        print("\tdid not generate error for incorrect input - invalid category")
        return

    try:
        test = News("world", "a", 5, 1)
    except TypeError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of TypeError - wrong excitement_rate type".format(type(e)))
        return
    else:
        print("\tdid not generate error for incorrect input - wrong excitement_rate type")
        return

    try:
        test = News("world", 1.1, 5, 1)
    except ValueError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of ValueError - wrong excitement_rate value".format(type(e)))
    else:
        print("\tdid not generate error for incorrect input - wrong excitement_rate value")
        return

    try:
        test = News("world", -0.5, 5, 1)
    except ValueError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of ValueError - wrong excitement_rate value".format(type(e)))
    else:
        print("\tdid not generate error for incorrect input - wrong excitement_rate value")
        return

    try:
        test = News("world", 0.5, 5.4, 1)
    except TypeError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of TypeError - wrong validity_length type".format(type(e)))
        return
    else:
        print("\tdid not generate error for incorrect input - wrong validity_length type")
        return

    try:
        test = News("world", 1.1, 15, 1)
    except ValueError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of ValueError - wrong validity_length value".format(type(e)))
    else:
        print("\tdid not generate error for incorrect input - wrong validity_length value")
        return

    try:
        test = News("world", 1.1, 0, 1)
    except ValueError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of ValueError - wrong validity_length value".format(type(e)))
    else:
        print("\tdid not generate error for incorrect input - wrong validity_length value")
        return

    try:
        test = News("world", 0.5, 5, 1.4)
    except TypeError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of TypeError - wrong created type".format(type(e)))
        return
    else:
        print("\tdid not generate error for incorrect input - wrong created type")
        return

    try:
        test = News("world", 1.1, 5, 1)
    except ValueError:
        pass
    except Exception as e:
        print("\tgenerated wrong type of error. Got {} insted of ValueError - wrong created value".format(type(e)))
    else:
        print("\tdid not generate error for incorrect input - wrong created value")
        return

    print("Great news (no pun intended)! News.check_data() passed all tests!")


def test_get_excitement():
    print("Testing News.get_excitement()...")

    with open('get_excitement_tests.pkl', 'rb') as f:
        test_list = pickle.load(f)

    for cat, er, valid, created, step, res in test_list:
        news_item = News(cat, er, valid, created)
        test_structure(news_item)

        try:
            stud_res = news_item.get_excitement(step)
        except Exception:
            print("\tNews.get_excitement() generated error for step {}".format(step))
            print("\tNews attributes: {}, {}, {}, {}".format(cat, er, valid, created))
            return

        if not isinstance(stud_res, float):
            print("\tWrong return type for News.get_excitement(). Expected {}, got {}".format(float, type(stud_res)))
            return

        if abs(stud_res - res) > 0.001:
            print("\tNews.get_excitement() returned wrong value: expected {}, got {}".format(res, stud_res))
            print("\tNews attributes: {}, {}, {}, {}; step {}".format(cat, er, valid, created, step))
            return

    print("Great news (no pun intended)! News.get_excitement() passed all tests!")


def main():
    test_check_data()
    print()

    test_get_excitement()
    print()


if __name__ == '__main__':
    main()
