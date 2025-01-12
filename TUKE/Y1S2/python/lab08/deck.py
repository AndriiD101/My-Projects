from random import choice

from card import Card, SUITS, VALUES


class Deck:
    def __init__(self):
        self.all_cards = list()
        self.used_cards = list()
        self.available_cards = list()

        for suit in SUITS:
            for value in VALUES:
                card = Card(suit, value)
                self.all_cards.append(card)
                self.available_cards.append(card)

    def draw_card(self, card=None):
        if card is None:
            card = choice(self.available_cards)
        self.used_cards.append(card)
        self.available_cards.remove(card)

        return card

    def generate_setup(self):
        player1_hand = list()
        player2_hand = list()
        flop = list()
        turn = list()

        player1_hand.append(self.draw_card())
        player1_hand.append(self.draw_card())
        player2_hand.append(self.draw_card())
        player2_hand.append(self.draw_card())

        flop.append(self.draw_card())
        flop.append(self.draw_card())
        flop.append(self.draw_card())

        turn.append(self.draw_card())

        return player1_hand, player2_hand, flop, turn
