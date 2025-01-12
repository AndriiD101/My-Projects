import random


def load_words(dataset_path):
    src_file = open(dataset_path)
    words = []
    for line in src_file.readlines():
        words.append(line[:-1])
    return words


def get_puzzle(word_list):
    return random.choice(word_list)


def is_game_finished(guess, puzzle):
    return guess.lower() == puzzle.lower()


def evaluate_guess(guess, puzzle):
    guess = guess.lower()
    result = []
    for idx, let in enumerate(guess):
        letter_info = (let, let in puzzle, let == puzzle[idx])
    return list()


def start_game(dataset_path):
    word_list = load_words(dataset_path)
    puzzle = get_puzzle(word_list)
    for count in range(6):
        guess = ""
        while not guess in word_list:
            guess=input("Enter your guess: ")
        print(evaluate_guess(guess, puzzle))
        if is_game_finished(guess, puzzle):
            print('you won!')
            break
        else:
            print('you lost')

    pass


if __name__ == '__main__':
    start_game("C:/Users/denys/OneDrive/Desktop/Programming/TUKE/S2/python/word_list.txt")
