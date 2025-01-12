import random

#1
def load_words(dataset_path):
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

#2
def get_puzzle(word_list):
    return (random.choice(word_list))

#3
def is_game_finished(guess, puzzle):
    return guess.lower() == puzzle.lower()

#4
def evaluate_guess(guess, puzzle):
    guess = guess.lower()
    result = []
    for idx, let in enumerate (guess):
        letter_info = (let, let in puzzle, let == guess [idx])
        result.append(letter_info)

    return result()

#5
def start_game(dataset_path):
    word_list = load_words(dataset_path)
    if len(word_list):
        print(puzzle)
    puzzle = get_puzzle(word_list)
    print("Welcome to the  Word Guesser!")
    for count in range (6):
        guess = ""
        while not guess in word_list:
            guess = input ("Enter your guess: ")
        print (evaluate_guess(guess, puzzle))
        if is_game_finished(guess, puzzle):
            print("You win")
            break
        else:
            print("You lose. The word i was looking for is" , puzzle)


if __name__ == '__main__':

    start_game("word_list.txt")
