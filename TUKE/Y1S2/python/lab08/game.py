from deck import Deck
from player import Player


class Game:
    def __init__(self):
        self.deck = Deck()

        self.player1 = Player()
        self.player2 = Player()

    def print_setup(self):
        print("PLAYER1", self.player1)
        print("PLAYER2", self.player2)

        print("FLOP", [str(card) for card in self.flop])
        print("TURN", [str(card) for card in self.turn])

    def prepare_setup(self):
        # generates and sets up all values:
        # player1 hand, player2 hand, flop, and turn
        p1_hand, p2_hand, flop, turn = self.deck.generate_setup()
        self.player1.deal_hand(p1_hand)
        self.player2.deal_hand(p2_hand)
        self.flop = flop
        self.turn = turn
        pass

    def find_winner(self, river):
        # finds the winner of the game given the river card
        # returns 1 if player1 wins
        # returns 0 in case of draw
        # return -1 if player2 wins
        pass

    def calculate_chances(self):
        # simulates all possible game outcomes given setup
        # returns chance of player1 winning, draw, and player1 winning
        return 0.0, 0.0, 0.0
