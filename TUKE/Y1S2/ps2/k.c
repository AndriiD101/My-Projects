#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>
#include "k.h"

void add_random_tile(struct game *game){
    int row, col;
    // find random, but empty tile
    do{
        row = rand() % SIZE;
        col = rand() % SIZE;
    }while(game->board[row][col] != ' ');

    // place to the random position 'A' or 'B' tile
    if(rand() % 2 == 0){
        game->board[row][col] = 'A';
    }else{
        game->board[row][col] = 'B';
    }
}

bool is_game_won(const struct game game){
    for(int rows = 0; rows < SIZE; rows++){
        for(int cols = 0; cols < SIZE; cols++){
            if(game.board[rows][cols] == 'K') return true;
        }
    }
    return false;
}

bool is_move_possible(const struct game game){
    //horizontal checking
    // from left to right
    for(int rows = 0; rows < SIZE; rows++){
        for(int cols = 0; cols < SIZE-1; cols++){
            if(game.board[rows][cols] == game.board[rows][cols+1] || game.board[rows][cols+1]==' ') return true;
        }
    }
    //from right to left
    for(int rows = 0; rows < SIZE; rows++){
        for( int cols = SIZE-1; cols > 0; cols--){
            if(game.board[rows][cols]  == game.board[rows][cols-1] || game.board[rows][cols-1]==' ') return true;
        }
    }
    //vertical checking
    //from top to down
    for(int rows = 0; rows<SIZE-1; rows++){
        for(int cols = 0; cols<SIZE; cols++){
            if(game.board[rows][cols] == game.board[rows+1][cols] || game.board[rows+1][cols] == ' ') return true;
        }
    }
    //down to top
    for(int rows = SIZE-1; rows>0; rows--){
        for(int cols = 0; cols<SIZE; cols++){
            if(game.board[rows][cols] == game.board[rows-1][cols] || game.board[rows-1][cols] == ' ') return true;
        }
    }
    return false;
}


double power(double base, int exponent) {
    if (exponent == 0)
        return 1;
    
    double result = 1;
    int i;
    for (i = 0; i < abs(exponent); ++i) {
        result *= base;
    }

    if (exponent < 0)
        return 1 / result;
    else
        return result;
}

bool update(struct game *game, int dy, int dx){
    bool flag=false;
    char alpha[]="ABCDEFGHIJK";
    flag = is_move_possible(*game);
    if (game && ((dy == 0 && (dx == -1 || dx == 1)) || ((dy == -1 || dy == 1) && dx == 0))){
        //move letters up
        if (dx == 0 && dy == -1) {
            int game_score = 0;
            // Shift blocks upwards
            for (int col = 0; col < SIZE; ++col) {
                for (int row = 0; row < SIZE; ++row) {
                    if (game->board[row][col] != ' ') {
                        for (int i = 1; i <= 3; ++i) {
                            if ((row - i) >= 0 && game->board[row - i][col] == ' ') {
                                game->board[row - i][col] = game->board[row - i + 1][col];
                                game->board[row - i + 1][col] = ' ';
                                flag = true;
                            }
                        }
                    }
                }
            }
            // Merge blocks and calculate points
            for (int col = 0; col < SIZE; ++col) {
                for (int row = 0; row < SIZE; ++row) {
                    if (game->board[row][col] != ' ') {
                        if ((row + 1) < SIZE && game->board[row][col] == game->board[row + 1][col]) {
                            for (int i = 0; i < strlen(alpha); ++i) {
                                if (game->board[row][col] == alpha[i]) {
                                    game->board[row][col] = alpha[i + 1];
                                    game_score = (i + 1) + 1;
                                    game_score = (int)power(2, game_score);
                                    flag = true;
                                    game->board[row + 1][col] = ' ';
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // Shift blocks upwards again
            for (int col = 0; col < SIZE; ++col) {
                for (int row = 0; row < SIZE; ++row) {
                    if (game->board[row][col] != ' ') {
                        for (int i = 1; i <= 3; ++i) {
                            if ((row - i) >= 0 && game->board[row - i][col] == ' ') {
                                game->board[row - i][col] = game->board[row - i + 1][col];
                                game->board[row - i + 1][col] = ' ';
                            }
                        }
                    }
                }
            }
            game->score += game_score;
        }
        //move letters down
        if (dx == 0 && dy == 1) {
            int game_score = 0;
            // Shift blocks downwards
            for (int col = 0; col < SIZE; ++col) {
                for (int row = SIZE - 1; row >= 0; --row) {
                    if (game->board[row][col] != ' ') {
                        for (int i = 1; i <= 3; ++i) {
                            if ((row + i) < SIZE && game->board[row + i][col] == ' ') {
                                game->board[row + i][col] = game->board[row + i - 1][col];
                                game->board[row + i - 1][col] = ' ';
                                flag = true;
                            }
                        }
                    }
                }
            }
            // Merge blocks and calculate points
            for (int col = 0; col < SIZE; ++col) {
                for (int row = SIZE - 1; row >= 0; --row) {
                    if (game->board[row][col] != ' ') {
                        if ((row - 1) >= 0 && game->board[row][col] == game->board[row - 1][col]) {
                            for (int i = 0; i < strlen(alpha); ++i) {
                                if (game->board[row][col] == alpha[i]) {
                                    game->board[row][col] = alpha[i + 1];
                                    game_score = (i + 1) + 1;
                                    game_score = (int)power(2, game_score);
                                    flag = true;
                                    game->board[row - 1][col] = ' ';
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // Shift blocks downwards again
            for (int col = 0; col < SIZE; ++col) {
                for (int row = SIZE - 1; row >= 0; --row) {
                    if (game->board[row][col] != ' ') {
                        for (int i = 1; i <= 3; ++i) {
                            if ((row + i) < SIZE && game->board[row + i][col] == ' ') {
                                game->board[row + i][col] = game->board[row + i - 1][col];
                                game->board[row + i - 1][col] = ' ';
                            }
                        }
                    }
                }
            }
            game->score += game_score;
        }
        //move letters right
        if (dx == 1 && dy == 0) {
            int game_score = 0;
            for (int row = 0; row < SIZE; ++row) {
                for (int col = SIZE; col >= 0; --col) {
                    if (game->board[row][col] != ' ') {
                        for (int i = 1; i <= 3; ++i) {
                            if ((col + i) < SIZE && game->board[row][col + i] == ' ') {
                                game->board[row][col + i] = game->board[row][col + i - 1];
                                game->board[row][col + i - 1] = ' ';
                                flag = true;
                            }
                        }
                    }
                }
            }
            // combine cols and calculate number
            for (int row = 0; row < SIZE; ++row) {
                for (int col = SIZE; col >= 0; --col) {
                    if (game->board[row][col] != ' ') {
                        if ((col - 1) >= 0 && game->board[row][col] == game->board[row][col - 1]) {
                            for (int i = 0; i < strlen(alpha); ++i) {
                                if (game->board[row][col] == alpha[i]) {
                                    game->board[row][col] = alpha[i + 1];
                                    game_score = (i + 1) + 1;
                                    game_score = (int)power(2, game_score);
                                    flag = true;
                                    game->board[row][col - 1] = ' ';
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // Shift blocks to the right again
            for (int row = 0; row < SIZE; ++row) {
                for (int col = SIZE; col >= 0; --col) {
                    if (game->board[row][col] != ' ') {
                        for (int i = 1; i <= 3; ++i) {
                            if ((col + i) < SIZE && game->board[row][col + i] == ' ') {
                                game->board[row][col + i] = game->board[row][col + i - 1];
                                game->board[row][col + i - 1] = ' ';
                            }
                        }
                    }
                }
            }
            game->score += game_score;
        }
        //move letters left
        if (dx == -1 && dy == 0) {
            int game_score = 0;
            // Shift blocks to the left
            for (int row = 0; row < SIZE; ++row) {
                for (int col = 0; col < SIZE; ++col) {
                    if (game->board[row][col] != ' ') {
                        for (int i = 1; i <= 3; ++i) {
                            if ((col - i) >= 0 && game->board[row][col - i] == ' ') {
                                game->board[row][col - i] = game->board[row][col - i + 1];
                                game->board[row][col - i + 1] = ' ';
                                flag = true;
                            }
                        }
                    }
                }
            }
            // Merge blocks and calculate points
            for (int row = 0; row < SIZE; ++row) {
                for (int col = 0; col < SIZE; ++col) {
                    if (game->board[row][col] != ' ') {
                        if ((col + 1) < SIZE && game->board[row][col] == game->board[row][col + 1]) {
                            for (int i = 0; i < strlen(alpha); ++i) {
                                if (game->board[row][col] == alpha[i]) {
                                    game->board[row][col] = alpha[i + 1];
                                    game_score = (i + 1) + 1;
                                    game_score = (int)power(2, game_score);
                                    flag = true;
                                    game->board[row][col + 1] = ' ';
                                    break;
                                }
                            }
                        }
                    }
                }
            }
            // Shift blocks to the left again
            for (int row = 0; row < SIZE; ++row) {
                for (int col = 0; col < SIZE; ++col) {
                    if (game->board[row][col] != ' ') {
                        for (int i = 1; i <= 3; ++i) {
                            if ((col - i) >= 0 && game->board[row][col - i] == ' ') {
                                game->board[row][col - i] = game->board[row][col - i + 1];
                                game->board[row][col - i + 1] = ' ';
                            }
                        }
                    }
                }
            }
            game->score += game_score;
        }
    }
    return flag;
}