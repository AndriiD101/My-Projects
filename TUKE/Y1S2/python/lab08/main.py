from game import Game


def main():
    game = Game()

    game.prepare_setup()
    game.print_setup()

    p1_chance, draw_chance, p2_chance = game.calculate_chances()

    print("P1 win:", p1_chance)
    print("DRAW:", draw_chance)
    print("P2 win:", p2_chance)


if __name__ == '__main__':
    main()
