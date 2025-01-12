#include <stdio.h>
#include <stdlib.h>
//#include "c4.h" 
#include <curses.h>
#include <unistd.h>


void initialize_board(int rows, int cols, char board[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            board[i][j] = '.';
        }
    }
}

void print_board(int rows, int cols, const char board[rows][cols]) {
    move(0,0);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j == 0) {
                attron(COLOR_PAIR(3));
                printw(" |");
                attron(COLOR_PAIR(3));
            }
            if (board[i][j] == 'X') {
                attron(COLOR_PAIR(1));
                printw(" %c", board[i][j]);
                attroff(COLOR_PAIR(1));
                attron(COLOR_PAIR(3));
                printw(" |");
                attron(COLOR_PAIR(3));
            }
            else if (board[i][j] == 'O') {
                attron(COLOR_PAIR(2));
                printw(" %c", board[i][j]);
                attroff(COLOR_PAIR(2));
                attron(COLOR_PAIR(3));
                printw(" |");
                attron(COLOR_PAIR(3));
            }
             else if (board[i][j] == '.') {
                attron(COLOR_PAIR(4));
                printw(" %c", board[i][j]);
                attroff(COLOR_PAIR(4));
                attron(COLOR_PAIR(3));
                printw(" |");
                attron(COLOR_PAIR(3));
            }
        }
       printw("\n");
    }
    printw(" +");
    for (int col = 0; col < cols; col++) {
        attron(COLOR_PAIR(3));
        printw("---+");
        attroff(COLOR_PAIR(3));
    }
    printw("\n");
    for (int col = 1; col <= cols; col++) {
        attron(COLOR_PAIR(5));
        printw("  %2d", col);
        attron(COLOR_PAIR(5));
    }
    printw("\n\n");
    refresh();
}

int is_valid_move(int rows, int cols, const char board[rows][cols], int col) {
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
    attron(COLOR_PAIR(6));
    while (1) {
        print_board(rows, cols, board);
        printw("Player %d, enter the column number (1-%d): ", player_turn, cols);
        refresh();
        scanw("%d", &col);
        col--;
        clear();
        //clear();
        if (!is_valid_move(rows, cols, board, col)) {
            attron(COLOR_PAIR(3));
            sleep(2);
            printw("Invalid column number or column is full. Please choose another column.");
            refresh();
            continue;
        }
        int row = drop_piece(rows, cols, board, col, player_piece);
        if (check_win(rows, cols, board, row, col, player_piece)) {
            print_board(rows, cols, board);
            attron(COLOR_PAIR(3));
            printw("Player %d wins!\n", player_turn);
            refresh();
            break;
        }
        player_turn = 3 - player_turn;
        player_piece = (player_piece == 'X') ? 'O' : 'X';

        if (is_board_full(rows, cols, board)) {
            print_board(rows, cols, board);
            attron(COLOR_PAIR(3));
            printw("The game is a tie!\n");
            break;
        }
    }
    printw("press any key");
    refresh();
    attroff(COLOR_PAIR(3));
    getch();
    refresh();
}

int main() {
    initscr();
    start_color();
    init_pair(1, COLOR_RED, COLOR_BLACK);
    init_pair(2, COLOR_GREEN, COLOR_BLACK);
    init_pair(3, COLOR_MAGENTA, COLOR_BLACK);
    init_pair(4, COLOR_YELLOW, COLOR_BLACK);
    init_pair(5, COLOR_WHITE, COLOR_BLACK);
    init_pair(6, COLOR_BLUE, COLOR_BLACK);
    int rows = 5;
    int cols = 5;
    clrtobot();
    play_connect4_game(rows, cols);
    clrtobot();
    move(0,0);
    clear();
    refresh();
    return 0;
    endwin();
}
