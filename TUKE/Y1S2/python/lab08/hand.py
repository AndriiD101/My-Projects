from copy import deepcopy


class Hand:
    def __init__(self, cards):
        self.cards = cards

    def __str__(self):
        str_repr = ""
        for card in self.cards:
            str_repr += "{} ".format(str(card))
        return str_repr

    def is_royal_flush(self):
        suit = set([card.suit for card in self.cards])
        if len(suit) != 1:
            return False, None

        must_values = [10, 11, 12, 13, 14]
        ordered_copy = deepcopy(self.cards)
        ordered_copy.sort(key=lambda c: c.value)
        values = [card.value for card in ordered_copy]
        if values != must_values:
            return False, None

        return True, (14,)

    def is_straight_flush(self):
        ordered_copy = deepcopy(self.cards)
        ordered_copy.sort(key=lambda c: c.value)
        suit = set([card.suit for card in self.cards])
        if len(suit) != 1:
            return False, None

        values = [card.value for card in ordered_copy]
        min_value = min(values)
        for val in range(min_value + 1, min_value + 5):
            if val not in values:
                return False, None

        return True, (max(values),)

    def is_four_of_a_kind(self):
        values = [card.value for card in self.cards]
        unique_values = set(values)
        if len(unique_values) != 2:
            return False, None

        for val in unique_values:
            if values.count(val) == 4:
                return True, (val,)

        return False, None

    def is_full_house(self):
        values = [card.value for card in self.cards]
        unique_values = set(values)
        if len(unique_values) != 2:
            return False, None

        vals = []
        card_counts = []
        for val in unique_values:
            card_counts.append(values.count(val))
            vals.append(val)

        if card_counts in [[3, 2], [2, 3]]:
            three_val = vals[card_counts.index(3)]
            two_val = vals[card_counts.index(2)]
            return True, (three_val, two_val)

        return False, None

    def is_flush(self):
        card_suits = [card.suit for card in self.cards]
        unique_suits = set(card_suits)

        suit_count = len(unique_suits)

        if suit_count == 1:
            return True, (max([card.value for card in self.cards]),)
        else:
            return False, None

    def is_straight(self):
        ordered_copy = deepcopy(self.cards)
        ordered_copy.sort(key=lambda c: c.value)
        values = [card.value for card in ordered_copy]
        min_value = min(values)
        for val in range(min_value + 1, min_value + 5):
            if val not in values:
                return False, None

        return True, (max(values),)

    def is_three_of_a_kind(self):
        values = [card.value for card in self.cards]
        unique_values = set(values)

        for val in unique_values:
            if values.count(val) == 3:
                return True, (val,)

        return False, None

    def is_two_pairs(self):
        values = [card.value for card in self.cards]
        unique_values = set(values)

        number_of_pairs = 0
        pair_vals = []
        for val in unique_values:
            if values.count(val) == 2:
                pair_vals.append(val)
                number_of_pairs += 1

        if number_of_pairs == 2:
            pair_vals.sort(reverse=True)
            return True, (pair_vals[0], pair_vals[1])

        return False, None

    def is_pair(self):
        values = [card.value for card in self.cards]
        unique_values = set(values)

        number_of_pairs = 0
        pair_vals = []
        for val in unique_values:
            if values.count(val) == 2:
                pair_vals.append(val)
                number_of_pairs += 1

        if number_of_pairs == 1:
            return True, (pair_vals[0],)

        return False, None

    def evaluate_hand(self):
        res = self.is_royal_flush()
        if res[0]:
            return 9, res[1]

        res = self.is_straight_flush()
        if res[0]:
            return 8, res[1]

        res = self.is_four_of_a_kind()
        if res[0]:
            return 7, res[1]

        res = self.is_full_house()
        if res[0]:
            return 6, res[1]

        res = self.is_flush()
        if res[0]:
            return 5, res[1]

        res = self.is_straight()
        if res[0]:
            return 4, res[1]

        res = self.is_three_of_a_kind()
        if res[0]:
            return 3, res[1]

        res = self.is_two_pairs()
        if res[0]:
            return 2, res[1]

        res = self.is_pair()
        if res[0]:
            return 1, res[1]

        return 0, None

    def compare_highest_card(self, other):
        unique_in_1 = sorted(
            [card for card in self.cards if card not in other.cards],
            key=lambda c: c.value, reverse=True)
        unique_in_2 = sorted(
            [card for card in other.cards if card not in self.cards],
            key=lambda c: c.value, reverse=True)

        for i in range(len(unique_in_1)):
            if unique_in_1[i].value > unique_in_2[i].value:
                return 1
            if unique_in_2[i].value > unique_in_1[i].value:
                return -1

        return 0

    def __eq__(self, other):
        if self.__class__ != other.__class__:
            raise TypeError("Cannot compare {} with {}".format(
                type(self), type(other)))

        code, vals = self.evaluate_hand()
        other_code, other_vals = other.evaluate_hand()

        if code != other_code:
            return False

        if code != 0:
            for my_val, other_val in zip(vals, other_vals):
                if my_val != other_val:
                    return False

        return self.compare_highest_card(other) == 0

    def __gt__(self, other):
        if self.__class__ != other.__class__:
            raise TypeError("Cannot compare {} with {}".format(
                type(self), type(other)))

        code, vals = self.evaluate_hand()
        other_code, other_vals = other.evaluate_hand()

        if code != other_code:
            return code > other_code
        elif code != 0:
            for my_val, other_val in zip(vals, other_vals):
                if my_val != other_val:
                    return my_val > other_val

        return self.compare_highest_card(other) == 1
