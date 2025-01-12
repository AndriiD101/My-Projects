SUITS = ['H', 'C', 'S', 'D']
VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


class Card:
    def __init__(self, suit, value):
        if suit not in SUITS:
            raise ValueError("Unknown card suit", suit)

        if value not in VALUES:
            raise ValueError("Unknown card value", value)

        self.suit = suit
        self.value = value

    def __str__(self):
        return "{}{}".format(self.suit, self.value)

    def __eq__(self, other):
        # compares two cards
        # two cards are the same if their suit and value matches
        if self.__class__ != other.__class__:
            raise TypeError("Cannot compare {} with {}".format(
                type(self), type(other)))
        return self.suit == other.suit and self.value == other.value

    def __gt__(self, other):
        # compares two cards
        # a card is bigger than the other if its value is higher
        if self.__class__ != other.__class__:
            raise TypeError("Cannot compare {} with {}".format(
                type(self), type(other)))

        return self.value > other.value


if __name__ == "__main__":
    card = Card("H", 7)
    print (card)
    