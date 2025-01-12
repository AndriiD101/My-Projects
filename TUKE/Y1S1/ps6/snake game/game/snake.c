#include <ncurses.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#define WIDTH 80
#define HEIGHT 24

int snakeX, snakeY, fruitX, fruitY, score;
int tailX[100], tailY[100];
int nTail;
int speed;
enum eDirection { STOP = 0, LEFT, RIGHT, UP, DOWN };
enum eDirection dir;
bool isLevelCompleted = false;
int currentLevel = 1;
void MainMenu();
void PrintAsciiArt();

int TopScore = 0;

void WriteTopScore() {
    FILE *file = fopen("top_score.txt", "w+");

    if (file != NULL) {
        fprintf(file, "%d\n", TopScore);
        fclose(file); 
    } else {
        printf("Error: Unable to open the file: top_score.txt\n"); 
    }
}

void ReadTopScore() {
    FILE *file = fopen("top_score.txt", "r");

    if (file != NULL) {
        fscanf(file, "%d", &TopScore);
        fclose(file); 
    } else {
        printf("Error: Unable to open the file: top_score.txt\n"); 
    }
}

void DisplayTopScore() {
    clear();
    PrintAsciiArt();

    attron(COLOR_PAIR(3));
    mvprintw(20, 25, "Highest Score: %d", TopScore);
    attroff(COLOR_PAIR(3));
    refresh();

    getch();
}

void PrintAsciiArt() {

    attron(COLOR_PAIR(3));
    mvprintw(4, 25, "Highest Score: %d", TopScore);
    attroff(COLOR_PAIR(3));
    refresh();

    attron(COLOR_PAIR(1)); // Set color to green
    mvprintw(7, 10, " $$$$$$\\  $$\\   $$\\  $$$$$$\\  $$\\   $$\\ $$$$$$$$\\ ");
    mvprintw(8, 10, "$$  __$$\\ $$$\\  $$ |$$  __$$\\ $$ | $$  |$$  _____|");
    mvprintw(9, 10, "$$ /  \\__|$$$$\\ $$ |$$ /  $$ |$$ |$$  / $$ |      ");
    mvprintw(10, 10, "\\$$$$$$\\  $$ $$\\$$ |$$$$$$$$ |$$$$$  /  $$$$$\\    ");
    mvprintw(11, 10, " \\____$$\\ $$ \\$$$$ |$$  __$$ |$$  $$<   $$  __|   ");
    mvprintw(12, 10, "$$\\   $$ |$$ |\\$$$ |$$ |  $$ |$$ |\\$$\\  $$ |      ");
    mvprintw(13, 10, " \\$$$$$$  |$$ | \\$$ |$$ |  $$ |$$ | \\$$\\ $$$$$$$$\\ ");
    mvprintw(14, 10, " \\______/ \\__|  \\__|\\__|  \\__|\\__|  \\__|\\________|");
    attroff(COLOR_PAIR(1)); // Turn off color
}

void InitColors() {
    start_color();
    init_pair(1, COLOR_GREEN, COLOR_BLACK); // Green color pair
    init_pair(2, COLOR_MAGENTA, COLOR_BLACK); // Snake
    init_pair(3, COLOR_RED, COLOR_BLACK);     // Fruit
}

void Setup() {
    clear();
    curs_set(0);
    srand(time(0));

    dir = RIGHT;
    snakeX = WIDTH / 4;
    snakeY = HEIGHT / 2;
    fruitX = (rand() % (WIDTH - 2)) + 1;
    fruitY = (rand() % (HEIGHT - 2)) + 1;
    score = 0;
    nTail = 0;
    speed = 150000;
    isLevelCompleted = false;
    InitColors();
}

void InitLevelThree() {
    int midWidth = WIDTH / 2;
    int midHeight = HEIGHT / 2;
    attron(COLOR_PAIR(1)); // Set color to green
    for (int i = 2; i <= midHeight; i++) {
        mvprintw(midHeight - i, midWidth - i, "#");
        mvprintw(midHeight + i, midWidth + i, "#");
        mvprintw(midHeight - i, midWidth + i, "#");
        mvprintw(midHeight + i, midWidth - i, "#");
    }
    attroff(COLOR_PAIR(1)); // Turn off color
}


void Draw() {
    clear();
    //attron(COLOR_PAIR(1));
    for (int i = 0; i < WIDTH + 2; i++){
        attron(COLOR_PAIR(1)); // Set color to green
        mvprintw(0, i, "+");
        attroff(COLOR_PAIR(1)); // Turn off color
    }
    for (int i = 1; i < HEIGHT + 1; i++) {
        for (int j = 0; j < WIDTH + 2; j++) {
            if (j == 0 || j == WIDTH + 1){
                attron(COLOR_PAIR(1)); // Set color to green
                mvprintw(i, j, "+");
                attroff(COLOR_PAIR(1)); // Turn off color
            }
            else if (i == snakeY && j == snakeX) {
                attron(COLOR_PAIR(2));
                mvprintw(i, j, "O");
                attroff(COLOR_PAIR(2));
            }
            else if (i == fruitY && j == fruitX) {
                attron(COLOR_PAIR(3));
                mvprintw(i, j, "*");
                attroff(COLOR_PAIR(3));
            }
            else {
                bool print = false;
                for (int k = 0; k < nTail; k++) {
                    if (tailX[k] == j && tailY[k] == i) {
                        attron(COLOR_PAIR(2));
                        mvprintw(i, j, "o");
                        attroff(COLOR_PAIR(2));
                        print = true;
                    }
                }
                if (!print) mvprintw(i, j, " ");
            }
        }
    }
    attron(COLOR_PAIR(1));
    for (int i = 0; i < WIDTH + 2; i++)
        mvprintw(HEIGHT + 1, i, "+");
    if (currentLevel == 2) {
        for (int i = 4; i < HEIGHT / 2 - 2; i++) {
            mvprintw(i, WIDTH / 4, "|");
            mvprintw(i, (3 * WIDTH) / 4, "|");
        }

        for (int i = 2; i < WIDTH - 2; i++) {
            if (i < WIDTH / 3 || i > 2 * WIDTH / 3) {
                mvprintw(HEIGHT / 4, i, "-");
                mvprintw((3 * HEIGHT) / 4, i, "-");
            }
        }
    }
    attroff(COLOR_PAIR(1));
    if (currentLevel == 3) {
        InitLevelThree();
    }
    //attroff(COLOR_PAIR(1));
    mvprintw(HEIGHT + 3, 0, "Score: %d", score);
    mvprintw(HEIGHT + 4, 0, "Use arrow keys to move the snake. 'n' for next level, 'x' to quit, 'q' to pause.");

    refresh();
}


void InitLevelTwo() {
    for (int i = 4; i < HEIGHT / 2 - 2; i++) {
        mvprintw(i, WIDTH / 4, "|");
        mvprintw(i, (3 * WIDTH) / 4, "|");
    }
    for (int i = 2; i < WIDTH - 2; i++) {
        if (i < WIDTH / 3 || i > 2 * WIDTH / 3) {
            mvprintw(HEIGHT / 4, i, "-");
            mvprintw((3 * HEIGHT) / 4, i, "-");
        }
    }
}


void ResetGame() {
    clear();
    nTail = 0;
    //score = 0;
    snakeX = WIDTH / 4;
    snakeY = HEIGHT / 2;
    fruitX = (rand() % (WIDTH - 2)) + 1;
    fruitY = (rand() % (HEIGHT - 2)) + 1;
    dir = RIGHT;
    isLevelCompleted = false;

    if (currentLevel == 1) {
        speed = 150000;
    } else if (currentLevel == 2) {
        speed = 100000;
        InitLevelTwo(); 
    }

    if (currentLevel == 3) {
        speed = 100000;
        InitLevelThree();
    }
}



void Input() {
    keypad(stdscr, TRUE);
    nodelay(stdscr, TRUE);

    int c = getch();
    switch(c) {
        case KEY_LEFT:
            if (dir != RIGHT) dir = LEFT;
            break;
        case KEY_RIGHT:
            if (dir != LEFT) dir = RIGHT;
            break;
        case KEY_UP:
            if (dir != DOWN) dir = UP;
            break;
        case KEY_DOWN:
            if (dir != UP) dir = DOWN;
            break;
        case 'q':
            nodelay(stdscr, FALSE);
            getch();
            nodelay(stdscr, TRUE);
            break;
        case 'x':
            dir = STOP;
            break;
        case 'n':
            if (isLevelCompleted) {
                ResetGame();
                nodelay(stdscr, TRUE);
            }
            break;

    }
}

void Logic() {
    int prevX = tailX[0];
    int prevY = tailY[0];
    int prev2X, prev2Y;
    tailX[0] = snakeX;
    tailY[0] = snakeY;
    for (int i = 1; i < nTail; i++) {
        prev2X = tailX[i];
        prev2Y = tailY[i];
        tailX[i] = prevX;
        tailY[i] = prevY;
        prevX = prev2X;
        prevY = prev2Y;
    }

    switch(dir) {
        case LEFT:
            snakeX--;
            break;
        case RIGHT:
            snakeX++;
            break;
        case UP:
            snakeY--;
            break;
        case DOWN:
            snakeY++;
            break;
        default:
            break;
    }

    if (snakeX == 0 || snakeX == WIDTH || snakeY == 0 || snakeY == HEIGHT) {
        dir = STOP;
    }

    for (int i = 0; i < nTail; i++) {
        if (tailX[i] == snakeX && tailY[i] == snakeY) {
            dir = STOP;
        }
    }
   if ((currentLevel == 1 && score >= 90) || 
        (currentLevel == 2 && score >= 140) || 
        (currentLevel == 3 && score >= 190)) {
        isLevelCompleted = true;
        if (currentLevel == 2) {
            mvprintw(HEIGHT + 6, 0, "Level 2 complete! Press 'n' to go to next level or 'x' to exit.");
        } else if (currentLevel == 3) {
            clear();
            const char* winMsg = "YOU WON";
            int msgLength = strlen(winMsg);
            int startX = (WIDTH - msgLength) / 2;
            mvprintw(HEIGHT / 2, startX, "%s", winMsg);
            mvprintw(HEIGHT / 2 + 1, (WIDTH / 2) - 20, "Press 'x' to continue.");
        }
        refresh();
        nodelay(stdscr, FALSE);
    }
    if (snakeX == fruitX && snakeY == fruitY) {
        score += 5;
        fruitX = (rand() % (WIDTH - 2)) + 1;
        fruitY = (rand() % (HEIGHT - 2)) + 1;
        nTail++;
        speed -= 5000;
        if (speed < 20000) {
            speed = 20000;
        }

       if (score >= 90 && !isLevelCompleted && currentLevel == 1) {
        isLevelCompleted = true;
        nodelay(stdscr, FALSE);
        mvprintw(HEIGHT + 6, 0, "Level 1 complete! Press 'n' to go to next level or 'x' to exit.");
        refresh();
    }
       int ch;
       while(isLevelCompleted) {
           ch = getch();
            if (ch == 'n' && currentLevel != 3) {
                currentLevel++;
                ResetGame();
                break;
            } else if (ch == 'x') {
                dir = STOP;
                break;
            }
        }
    }

    if (isLevelCompleted) {
        if (snakeY == HEIGHT / 4 || snakeY == (3 * HEIGHT) / 4) {
            if (snakeX > 1 && snakeX < WIDTH) {
                dir = STOP;
            }
        }
        if (snakeX == WIDTH / 4 || snakeX == (3 * WIDTH) / 4) {
            if (snakeY > 0 && snakeY < HEIGHT) {
                dir = STOP;
            }
        }
    }
    if (currentLevel == 2) {
        if ((snakeX == WIDTH / 4 || snakeX == (3 * WIDTH) / 4) && (snakeY >= 4 && snakeY <= HEIGHT / 2 - 2)) {
            dir = STOP;
        }
        if ((snakeY == HEIGHT / 4 || snakeY == (3 * HEIGHT) / 4) && (snakeX > 1 && snakeX < WIDTH - 1)) {
            if (!(snakeX >= WIDTH / 3 && snakeX <= 2 * WIDTH / 3)) {
                dir = STOP;
            }
        }
    }
    if (currentLevel == 3) {
        int midWidth = WIDTH / 2;
        int midHeight = HEIGHT / 2;
        for (int i = 2; i <= midHeight; i++) {
            if ((snakeX == midWidth - i && snakeY == midHeight - i) ||
                (snakeX == midWidth + i && snakeY == midHeight + i) ||
                (snakeX == midWidth + i && snakeY == midHeight - i) ||
                (snakeX == midWidth - i && snakeY == midHeight + i)) {
                dir = STOP;
                break;
            }
        }
    }


int ch;
    while(isLevelCompleted) {
        ch = getch();
        if (ch == 'n') {
            currentLevel = (currentLevel == 2) ? 3 : 2;
            ResetGame();
            break;
        } else if (ch == 'x') {
            dir = STOP;
            break;
        }
    }
}

void SelectMap() {
    int maps[] = {1, 2, 3};
    int choice;
    int highlight = 0;
    int num_options = 4;

    while (1) {
        clear();
        PrintAsciiArt();
        // Set color pair for green text
        attron(COLOR_PAIR(1));

        // Draw the bigger green frame
        for (int i = 0; i < 72; i++) {
            mvprintw(1, i, "+");
            mvprintw(25, i, "+");
        }
        for (int i = 2; i < 25; i++) {
            mvprintw(i, 0, "+");
            mvprintw(i, 71, "+");
        }
        attroff(COLOR_PAIR(1));


        for (int i = 0; i < num_options; i++) {
            if (i == highlight) {
                attron(A_REVERSE);
            }
            switch (i) {
                case 0: mvprintw(16, 25, "Map 1"); break;
                case 1: mvprintw(18, 25, "Map 2"); break;
                case 2: mvprintw(20, 25, "Map 3"); break;
                case 3: mvprintw(22, 25, "Back to Menu"); break;
            }
            attroff(A_REVERSE);
        }

        choice = getch();

        switch (choice) {
            case KEY_UP:
                if (highlight > 0) highlight--;
                break;
            case KEY_DOWN:
                if (highlight < num_options - 1) highlight++;
                break;
            case 10:
                if (highlight < 3) {
                    currentLevel = maps[highlight];
                    ResetGame(); 
                    return;
                } else if (highlight == 3) {
                    return;
                }
                break;
            default:
                break;
        }
    }
}

void SelectSpeed() {
    int speeds[] = {150000, 100000, 50000};
    int choice;
    int highlight = 0;
    int num_options = 4;

    keypad(stdscr, TRUE);

    while(1) {
        clear();
        PrintAsciiArt();
        // Set color pair for green text
        attron(COLOR_PAIR(1));

        // Draw the bigger green frame
        for (int i = 0; i < 72; i++) {
            mvprintw(1, i, "+");
            mvprintw(25, i, "+");
        }
        for (int i = 2; i < 25; i++) {
            mvprintw(i, 0, "+");
            mvprintw(i, 71, "+");
        }
        attroff(COLOR_PAIR(1));

        for(int i = 0; i < num_options; i++) {
            if(i == highlight)
                attron(A_REVERSE);
            switch (i) {
                case 0: mvprintw(16, 26, "Speed 150000"); break;
                case 1: mvprintw(18, 26, "Speed 100000"); break;
                case 2: mvprintw(20, 26, "Speed 50000"); break;
                case 3: mvprintw(22, 26, "Back to menu"); break;
            }
            attroff(A_REVERSE);
        }

        choice = getch();

        switch(choice) {
            case KEY_UP:
                if(highlight > 0)
                    highlight--;
                break;
            case KEY_DOWN:
                if(highlight < num_options - 1)
                    highlight++;
                break;
            case 10:
            case KEY_ENTER:
                if(highlight < 3) {
                    speed = speeds[highlight];
                }
                clear();
                return;
            default:
                break;
        }
    }
}

void MainMenu() {
    int choice;
    int highlight = 0;
    int num_options = 4; // Select Map, Select Speed, Start Game, Exit Game

    while (1) {
        clear();
        attron(COLOR_PAIR(1));
        PrintAsciiArt(); // Display ASCII art
        attroff(COLOR_PAIR(1));

        // Set color pair for green text
        attron(COLOR_PAIR(1));

        // Draw the bigger green frame
        for (int i = 0; i < 72; i++) {
            mvprintw(1, i, "+");
            mvprintw(25, i, "+");
        }
        for (int i = 2; i < 25; i++) {
            mvprintw(i, 0, "+");
            mvprintw(i, 71, "+");
        }
        attroff(COLOR_PAIR(1));
        // Display menu options in green
        for (int i = 0; i < num_options; i++) {
            if (i == highlight) {
                attron(A_REVERSE); // Highlight the current choice
            }
            switch (i) {
                case 0: mvprintw(16, 26, "Select map"); break;
                case 1: mvprintw(18, 25, "Select Speed"); break;
                case 2: mvprintw(20, 26, "Start Game"); break;
                case 3: mvprintw(22, 26, "Exit  Game"); break;
            }
            attroff(A_REVERSE);
        }

        //attroff(COLOR_PAIR(1)); // Turn off color pair

        choice = getch();

        switch (choice) {
            case KEY_UP:
                if (highlight > 0) highlight--;
                break;
            case KEY_DOWN:
                if (highlight < num_options - 1) highlight++;
                break;
            case 10: // Enter key
                if (highlight == 0) {
                    SelectMap();
                } else if (highlight == 1) {
                    SelectSpeed();
                } else if (highlight == 2) {
                    return; // Start the game
                } else if (highlight == 3) {
                    endwin(); // Close ncurses
                    exit(0); // Exit the program
                }
                break;
            default:
                break;
        }
    }
}


int main(int argc, char **argv) {
    ReadTopScore();
    if(argc == 2)
        if(strcmp("-reset_score", argv[1]) == 0){
            TopScore = 0;
            WriteTopScore();
        }
        else{
            printf("Wrong argument passed: %s\n", argv[1]);
            return 1;
        }
    initscr();
    Setup();
    noecho();
    cbreak();
    keypad(stdscr, TRUE);

    while (true) {
        MainMenu();

        while (dir != STOP) {
            Draw();
            Input();
            Logic();
            usleep(speed);
        }
        if(score>TopScore){
            TopScore = score;
            DisplayTopScore();
            WriteTopScore();
        }
        getch(); 
        endwin(); 

        printf("Final Score: %d\n", score);

        //printf("Play again? (y/n): ");
        //int choice = getchar();
        //if (choice == 'n' || choice == 'N') {
        //    WriteTopScore();
            return 0;
        }
    }

