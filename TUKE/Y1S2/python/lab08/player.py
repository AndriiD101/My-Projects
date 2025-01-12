from hand import Hand
from itertools import combinations

class Player:
    def __init__(self):
        self.hand = list()

    def __str__(self):
        str_repr = ""
        for card in self.hand:
            str_repr += "{} ".format(str(card))
        return str_repr

    def deal_hand(self, hand):
        self.hand = hand

    def get_all_combinations(self, flop, turn, river):
        # returns a list of all combinations of the given cards
        # each element in the list is of type Hand
        # Combine the flop, turn, and river into a single list
        community_cards = flop + [turn] + [river]
        
        # Generate all possible combinations of 5 cards from the 7 available
        all_five_card_combinations = list(combinations(community_cards, 5))
        
        # Convert each combination into a Hand object
        all_hands = [Hand(list(combination)) for combination in all_five_card_combinations]
        
        return all_hands
        return

    def get_best_hand(self, flop, turn, river):
        # returns the strongest hand from the available cards
        # the return type is Hand
        return max(self.get_all_combinations(flop, turn, river))
