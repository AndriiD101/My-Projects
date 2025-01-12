#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include "k.h"

#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "k.h"
#include "hof.h"
#include "ui.h"

void print_board(struct game *g) {
  printf("\n");
    for (int i = 0; i < 4; i++) {
      printf("|");
        for (int j = 0; j < 4; j++) {
            printf("%c|", g->board[i][j]);
        }
        printf("\n");
    }
}

int main(){
  // struct game game = {
  //     .board = {
  //         {'D', ' ', ' ', ' '},
  //         {'D', 'B', ' ', ' '},
  //         {' ', 'B', 'C', ' '},
  //         {' ', ' ', ' ', ' '}
  //     },
  //     .score = 0
  // };

  // printf("is won: %d\n", is_game_won(game));
  // printf("is move possible: %d\n", is_move_possible(game));
  // printf("update %d", update(&game, 0, -1));
  // print_board(&game);
  // printf("\n");
  // printf("%d", game.score);
struct player array[10];
struct player player = {
    .name = "john",
    .score = 400
};
int size = load(array);
printf("%d\n\n", load(array));
printf("%d\n\n", save(array, 10));
printf("%d\n\n", add_player(array, &size, player));
  // add_random_tile(&game);
  // render(game);
}


