#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "c4.h"

void initialize_board(int rows, int cols, char board[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            board[i][j] = '.';
        }
    }
}

void print_board(int rows, int cols, const char board[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j == 0) {
                printf(" |");
            }
            printf(" %c |", board[i][j]);
        }
        printf("\n");
    }
    printf(" +");
    for (int col = 0; col < cols; col++) {
        printf("---+");
    }
    printf("\n");
    for (int col = 1; col <= cols; col++) {
        printf("  %2d", col);
    }
    printf("\n\n");
}

int is_valid_move(int rows, int cols, const char board[rows][cols], int col) {
    int colik = col-1;
    if (colik < 0 || colik >= cols || board[0][colik] == 'X' || board[0][colik] == 'O') {
        return 0;
    }

    for (int row = 0; row<rows; row++) {
        if (board[row][colik] == '.') {
            return 1;
        }
    }
    return 0;
}

int is_valid_move_c4(int rows, int cols, const char board[rows][cols], int col) {
    int colik = col;
    if (colik < 0 || colik >= cols || board[0][colik] == 'X' || board[0][colik] == 'O') {
        return 0;
    }

    for (int row = 0; row<rows; row++) {
        if (board[row][colik] == '.') {
            return 1;
        }
    }
    return 0;
}


int drop_piece(int rows, int cols, char board[rows][cols], int col, char player_piece) {
    int colik = col-1;
    if (colik < 0 || colik >= cols) {
        return -1;
    }
    for(int row = rows -1; row >= 0; row--) {
        if(board[row][colik] == '.') {
            board[row][colik] = player_piece;
            return row;
        }
    }
    return -1;
}

int drop_piece_c4(int rows, int cols, char board[rows][cols], int col, char player_piece) {
    int colik = col;
    if (colik < 0 || colik >= cols) {
        return 0;
    }
    for(int row = rows -1; row >= 0; row--) {
        if(board[row][colik] == '.') {
            board[row][colik] = player_piece;
            return row;
        }
    }
    return 0;
}

int check_vertical(int rows, int cols, const char board[rows][cols], int row, int col, char player_piece) {
    int count = 0;
    int column = col;
    for (int i = 0; i < rows; i++) {
        if (board[i][column] == player_piece) {
            count++;
            if (count == 4) {
                return 1;
            }
        } else {
            count = 0;
        }
    }
    return 0;
}

int check_horizontal(int rows, int cols, const char board[rows][cols], int row, int col, char player_piece) {
    int count = 0;
    int l_row = row;
    for (int j = 0; j < cols; j++) {
        if (board[l_row][j] == player_piece) {
            count++;
            if (count == 4) {
                return 1;
            }
        } else {
            count = 0;
        }
    }
    return 0;
}

int check_diagonal(int rows, int cols, const char board[rows][cols], int row, int col, char player_piece) {
    //left-down to right-upper
    int l_row = row;
    int l_col = col;

    for (int r = l_row, c = l_col; r >= 0 && r < rows && c >= 0 && c < cols; r++, c--) {
        if (board[r][c] != player_piece) {
            break;
        }
        l_row++;
        l_col--;
    }

    int count = 0;
    // Diagonal counting from right-upper to left-down
    for (int r = l_row - 1, c = l_col + 1; r >= 0 && c < cols; r--, c++) {
        if (board[r][c] != player_piece) {
            break;
        }
        count++;
    }

    if (count >= 4) {
        return 1;
    }

    //left-upper to right-down
    l_row = row;
    l_col = col;

    for (int r = l_row, c = l_col; r >= 0 && r < rows && c >= 0 && c < cols; r--, c--) {
        if (board[r][c] != player_piece) {
            break;
        }
        l_row--;
        l_col--;
    }

    count = 0;
    // Diagonal counting from left-upper to right-down
    for (int r = l_row + 1, c = l_col + 1; r < rows && c < cols; r++, c++) {
        if (board[r][c] != player_piece) {
            break;
        }
        count++;
    }

    if (count >= 4) {
        return 1;
    }

    return 0;
}

// int check_diagonal(int rows, int cols, const char board[rows][cols], int row, int col, char player_piece) {
    // for (int i = 0; i < rows - 3; i++) {
    //     for (int j = 0; j < cols - 3; j++) {
    //         int count = 0;
    //         for (int k = 0; k < 4; k++) {
    //             if (board[i + k][j + k] == player_piece) {
    //                 count++;
    //             } else {
    //                 count = 0;
    //             }
    //             if (count == 4) {
    //                 return 1;
    //             }
    //         }
    //     }
    // }

    // // Check diagonals from bottom left to top right
    // for (int i = rows - 1; i >= 3; i--) {
    //     for (int j = 0; j < cols - 3; j++) {
    //         int count = 0;
    //         for (int k = 0; k < 4; k++) {
    //             if (board[i - k][j + k] == player_piece) {
    //                 count++;
    //             } else {
    //                 count = 0;
    //             }
    //             if (count == 4) {
    //                 return 1;
    //             }
    //         }
    //     }
    // }

//     return 0;
// }

int check_win(int rows, int cols, const char board[rows][cols], int row, int col, char player_piece) {
    return (check_vertical(rows, cols, board, row, col, player_piece) ||
            check_horizontal(rows, cols, board, row, col, player_piece) ||
            check_diagonal(rows, cols, board, row, col, player_piece));
}


int is_board_full(int rows, int cols, const char board[rows][cols]) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (board[i][j] == '.') {
                return 0;
            }
        }
    }
    return 1;
}

void play_connect4_game(int rows, int cols) {
    char board[rows][cols];
    int col;

    initialize_board(rows, cols, board);
    int player_turn = 1;
    char player_piece = 'X';

    while (1) {
	system("clear");
        print_board(rows, cols, board);
        printf("Player %d, enter the column number (1-%d): ", player_turn, cols);
        scanf("%d", &col);
        // Convert column number to index
        col--;

        // Check if the selected column is valid
        if (!is_valid_move_c4(rows, cols, board, col)) {
            printf("Invalid column number or column is full. Please choose another column.\n");
            sleep(1);
            continue;
        }

        // Drop the piece into the selected column
        int row = drop_piece_c4(rows, cols, board, col, player_piece);

        // Check for a winning move
        if (check_win(rows, cols, board, row, col, player_piece)) {
            print_board(rows, cols, board);
            printf("Player %d wins!\n", player_turn);
            return;
        }

        // Switch players
        player_turn = 3 - player_turn;
        player_piece = (player_piece == 'X') ? 'O' : 'X';

        // Check if the board is full
        if (is_board_full(rows, cols, board)) {
            print_board(rows, cols, board);
            printf("The game is a tie!\n");
            return;
        }
    }
}
/*int main() {
    int rows = 4;
    int cols = 7;
    // char board[rows][cols];
    // initialize_board(rows, cols, board);
    // board[4][4] = 'X';
    // board[3][4] = 'O';
    // // board[5][3] = 'O';
    // print_board(rows, cols, board);
    play_connect4_game(rows, cols);
    // printf("%d ", drop_piece(5, 5, board, 5, 'O'));
    // printf("%d", is_valid_move(5, 7, board, 7));
    return 0;
}*/
