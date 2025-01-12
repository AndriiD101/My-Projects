from copy import deepcopy
import random
import string


PLAYER_KNOWLEDGE = [(letter, None, -1)
                    for letter in string.ascii_lowercase]


class Puzzle:
    def __init__(self, word):
        self.puzzle = word.lower()

    def is_game_finished(self, guess):
        return guess.lower() == self.puzzle

    def evaluate_guess(self, guess):
        return list()


class Game:
    def __init__(self, dataset_path):
        self.word_list = self.load_words(dataset_path)
        self.puzzle = None

    def load_words(self, dataset_path):
        words = list()
        try:
            src_file = open(dataset_path, 'r')
        except FileNotFoundError:
            print("The file does not exist.")
            return []
        words = []
        for line in src_file.readlines():
            try:
                if len(line) == 6:
                    word = line[: -1]
                    if word.isalpha():
                        words.append(word.lower)
                    else:
                        raise ValueError("Word contains non-letter characters")
            except ValueError:
                pass
        return words

    def generate_puzzle(self):
        self.puzzle = Puzzle(random.choice(self.word_list))

    def start_game(self):
        self.generate_puzzle()
        print("Welcome to the  Word Guesser!")
        for count in range (6):
            guess = ""
            while guess not in self.word_list:
                guess = input ("Enter your guess: ")
            result = self.evaluate_guess(guess)
            print (result)
            if self.puzzle.is_game_finished(guess):
                print("You win")
                break
            else:
                print("You lose. The word i was looking for is" , self.puzzle.puzzle)

    def bot_game(self, bot):
        pass


class Bot:
    def __init__(self, word_list):
        self.reset(word_list)

    def reset(self, word_list):
        self.possibles = word_list
        self.knowladge = deepcopy(PLAYER_KNOWLEDGE)

    def get_player_guess(self):
        to_delete = list()
        for letter, contains, pos in self.knowladge:
            for word in self.possibles:
                if contains  is True and letter not in word:
                    to_delete.append(word)
                if contains in False and letter in word:
                    to_delete.append(word)
                if pos != -1 and word[pos] != letter:
                    to_delete.append(word)
        for word in to_delete:
            self.possibles.remove(word)
        return random.choice(self.possibles)

    def process_result(self, result):
        for idx, (letter, contains, pos) in enumerate(result):
            letter_info_idx = string.ascii_letters.find(letter)
            if pos:
                self.knowladges[letter_info_idx] = (letter, contains, idx)
            else:
                self.knowladgs[letter_info_idx] = (letter, contains, -1)


if __name__ == '__main__':
    wordle = Game("word_list.txt")
    wordle.start_game()
