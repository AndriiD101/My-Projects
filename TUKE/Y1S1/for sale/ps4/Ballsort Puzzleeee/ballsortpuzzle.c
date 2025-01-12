#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include "ballsortpuzzle.h"

#define ROWS 4
#define COLUMNS 6

void shuffle(const int rows, const int columns, char field[rows][columns], int xy[2]){
    // vygeneruje nahodnu poziciu
    int x, y;
    x = rand() % (columns-1);
    y = rand() % (rows-1);
    xy[0] = x;
    xy[1] = y;
    if(field[y][x] == ' '){
        shuffle(rows, columns, field, xy);
    }
}

void check_one(const int rows, const int columns, char field[rows][columns], int xy[2], int i, int j){
    // skontroluje iba jeden stlpec, ci je cely rovnaky
    int x, y;
    char temp;
    if(field[i][j] == field[i+1][j]){
        shuffle(rows, columns, field, xy);
        x = xy[0];
        y = xy[1];
        temp = field[i][j];
        field[i][j] = field[y][x];
        field[y][x] = temp;
        check_one(rows, columns, field, xy, i, j);
    }
}

bool check(const int rows, const int columns, char field[rows][columns]){
    // porovnava 2 nad sebou, spodne uz nekontroluje
    int rovnake = 0;
    for(int i = 0; i < columns; i++){
        for(int j = 0; j < rows-1; j++){
            if(field[j][i] == field[j+1][i]){
                rovnake++;
            }
        }
        if(rovnake == columns*(rows-1)){
            printf("Congratulations! You won!\n");
            return true;
        }
    }
    return false;
}

void generator(const int rows, const int columns, char field[rows][columns]){
    srand(time(0));
    char special_chars[] = {'@', '+', '^', '*', '#', '$', '>', '~', '<', '%'};
    int xy[2];
    int x, y = 0;
    char temp;
    int tempy = 0;

    for(int i = 0; i < rows; i++){
        if(i == 0){
            // generacia prvych medzier
            for(int j = 0; j < 2; j++){
                tempy = x;
                x = rand() % columns;
                if(j == 1 && x == tempy){
                    j--;
                    continue;
                }
                field[0][x] = ' ';
            }
            continue;
        }
    }
    for(int j = 0; j < rows; j++){
        for(int i = 0; i < columns; i++){
            // generacia medzier
            if(field[0][i] == ' '){
                field[j][i] = ' ';
            // generacia znakov
            } else {
                field[j][i] = special_chars[i];
            }
        }
    }
    // zamiesanie znakov
    for(int j = 0; j < columns; j++){
        for(int i = 0; i < rows; i++){
            if(field[i][j] == ' '){continue;}
            shuffle(rows, columns, field, xy);
            x = xy[0];
            y = xy[1];
            temp = field[i][j];
            field[i][j] = field[y][x];
            field[y][x] = temp;
        }
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < columns; j++){
            if(field[i][j] == ' '){
                continue;
            }
            check_one(rows, columns, field, xy, i, j);
        }
    }
}

void game_field(const int rows, const int columns, char field[rows][columns]){
    printf("\n");
    int tmp = 0;
    for(int j = 0; j < (rows+2); j++){
        for(int i = 0; i < (columns*4+3); i++){
            // vytlaci prvy stlpec (cisla)
            if(i == 0){
                if(j < rows){
                    if(j+1 < 10){
                        printf(" %d", j+1);
                    } else {
                        printf("%d", j+1);
                    }
                    continue;
                } else {
                    printf("%c%c", ' ', ' ');
                    continue;
                }
            }
            // vytlaci druhy stlpec (medzery)
            if(i == 1){
                printf("%c", ' ');
                continue;
            }
            // medzerove stlpce
            if((i%2) == 1){
                if(j != rows){
                    printf("%c", ' ');
                    continue;
                } else if (j == rows){
                    if(tmp == 0){
                        printf("%c", '\\');
                        tmp = 1;
                        continue;
                    } else if(tmp == 1){
                        printf("%c", '/');
                        tmp = 0;
                        continue;
                    }
                }
            }
            // palickove stlpce
            if((i%4) == 2){
                if(j < rows){
                    printf("%c", '|');
                    continue;
                } else if(j == rows){
                    printf("%c", '^');
                    continue;
                } else {
                    printf("%c", ' ');
                    continue;
                }
            }
            // znakove stlpce
            if((i%4) == 0){
                if(j < rows){
                    printf("%c", field[j][i/4-1]);
                    continue;
                } else if(j == rows){
                    printf("%c", '_');
                    continue;
                } else if(j > rows){
                    printf("%d", i/4);
                    continue;
                }
            }
        }
        printf("\n");
    }
}

void down_possible(const int rows, const int columns, char field[rows][columns], int x, int y){
    char tmp;
    int tempy;
    int err = 0;
    x--;
    y--;
    for(int i = 0; i < rows; i++){
        if(x > columns - 1 || x < 0 || y > columns - 1 || y < 0){
            printf("Out of bounds!\n\n");
            err = 1;
            break;
        }
        if(field[i][x] == ' ' && i == rows - 1){
            printf("Can't move nothing!\n\n");
            err = 1;
            break;
        }
        if(field[i][x] == ' '){continue;}
        tmp = field[i][x];
        tempy = i;
        field[i][x] = ' ';
        break;
    }
    for(int i = 0; i < rows+1; i++){
        if(err == 1){
            err = 0;
            break;
        }
        if(field[i][y] == ' '){continue;}
        if(i != rows && tmp != field[i][y]){
            printf("Characters must be same!\n\n");
            field[tempy][x] = tmp;
            break;
        }
        if(field[i][y] != ' ' && i == 0){
            printf("Can't move to a full bottle!\n\n");
            field[tempy][x] = tmp;
            break;
        }
        field[i-1][y] = tmp;
        break;
    }
}

void ball_sort_puzzle(){
    char tmp;
    int rows = ROWS;
    int columns = COLUMNS;
    int x = 0;
    int y = 0;
    
    printf("Do you want to set the width and height for your game? (Y/n) ");
    scanf("%c", &tmp);
    if(tmp == 'Y' || tmp == 'y'){
        printf("Height (2-10): ");
        scanf("%d", &rows);
        while(rows > 10 || rows < 2) {
            printf("Height should be at least 2 and at most 10!\n");
            printf("Height: ");
            scanf("%d", &rows);
        }
        printf("Width (4-10): ");
        scanf("%d", &columns);
        while(columns > 10 || columns < 4) {
            printf("Width should be at least 4 and at most 10!\n");
            printf("Width: ");
            scanf("%d", &columns);
        }
    }

    int c = 0;
    char field[rows][columns];
    generator(rows, columns, field);
    game_field(rows, columns, field);
    printf("\n");
    while(check(rows, columns, field) == false){
        printf("Move from: ");
        // discards everything except numbers
        while((c = getchar()) != '\n');
        scanf("%d", &x);
        printf("Move to: ");
        while((c = getchar()) != '\n');
        scanf("%d", &y);
        down_possible(rows, columns, field, x, y);
        game_field(rows, columns, field);
    }
}
