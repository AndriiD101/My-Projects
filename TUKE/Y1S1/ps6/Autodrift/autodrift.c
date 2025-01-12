#define _POSIX_C_SOURCE 200201L
#define ARRAY_SIZE(a) (sizeof(a) / sizeof(a[0]))
#define CTRLD 4
// change if you want different number of total levels (max. 5)
#define LEVELS 5
// change if you want a wider road (if you get segmentation fault, change to lower number)
#define PRECHOD 15
// change if you want different speed (slowest is 0.5, fastest 0.07)
#define SPEED 0.1
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <ncurses.h>
#include <time.h>
#include <unistd.h>

int menuwin();
void draw(char prechod_arr[LINES][COLS], int level);
int game(char prechod_arr[LINES][COLS], int level);

struct timespec ts = {
    .tv_sec = 0,                  // nr of secs
    .tv_nsec = 0.001 * 10000000L  // nr of nanosecs
};

int main(){
    srand(time(NULL));

    // ncurses boiler
    initscr();
    cbreak();
    noecho();
    keypad(stdscr, TRUE);
    curs_set(FALSE);
    nodelay(stdscr, TRUE);

    // colors start
    start_color();
    use_default_colors();
    init_pair(1, COLOR_RED, -1);
    init_pair(2, COLOR_YELLOW, -1);
    init_pair(3, COLOR_GREEN, -1);

    char prechod_arr[LINES][COLS];
    int level = 1;
    int gameint = 1;

    mvprintw(LINES/10, COLS/2-10, " W E L C O M E  T O ");
    
    // starts menu window and assings its return value to level
    menu_again:
   
    attron(COLOR_PAIR(2));
    mvprintw(LINES/10, COLS/2-32, "                                                                  ");
    mvprintw(LINES/10+1, COLS/2-32, " _______       _____            ________       _________________  ");
    mvprintw(LINES/10+2, COLS/2-32, " ___    |___  ___  /______      ___  __ \\_________(_)__  __/_  /_ ");
    mvprintw(LINES/10+3, COLS/2-32, " __  /| |  / / /  __/  __ \\     __  / / /_  ___/_  /__  /_ _  __/ ");
    mvprintw(LINES/10+4, COLS/2-32, " _  ___ / /_/ // /_ / /_/ /     _  /_/ /_  /   _  / _  __/ / /_   ");
    mvprintw(LINES/10+5, COLS/2-32, " /_/  |_\\__,_/ \\__/ \\____/      /_____/ /_/    /_/  /_/    \\__/   ");
    mvprintw(LINES/10+6, COLS/2-32, "                                                                  ");
    attroff(COLOR_PAIR(2));

    level = menuwin();
    if(level == 0){
        return EXIT_SUCCESS;
    }

    // game loop
    for(int level_idx = level; level_idx <= LEVELS; level_idx++){
        
        // draws the world for this level
        draw(prechod_arr, level_idx);
    
        // draw the level sign
        WINDOW *level_sign = newwin(3, 17, 4, 7);
        box(level_sign, 0, 0);
        mvwprintw(level_sign, 1, 2, "L E V E L : %d", level_idx);
        refresh();
        wrefresh(level_sign);
        
        // game
        gameint = game(prechod_arr, level_idx);
        if(gameint == 0){
            return EXIT_SUCCESS;
            break;
        } else if(gameint == 2){
            goto menu_again;
            break;
        }

        // end the game after the last level
        if(level_idx == LEVELS){
            mvprintw(LINES/2-1, COLS/2-16, "                                         ");
            mvprintw(LINES/2, COLS/2-16, " CONGRATULATIONS!!! YOU'VE WON THE GAME! ");
            mvprintw(LINES/2+1, COLS/2-16, "                                         ");
            refresh();
            sleep(2);
            goto menu_again;
        }

        // end of a level
        mvprintw(LINES/2-1, COLS/2-19, "                                       ");
        mvprintw(LINES/2, COLS/2-19, " Congratulations! You passed level %d! ", level_idx);
        mvprintw(LINES/2+2, COLS/2-19, "                                       ");
        refresh();
        sleep(2);
    }
}

int menuwin(){
    
    // creates new menu window
    WINDOW *menu = newwin(LINES/3, COLS/3, LINES/3, COLS/3);
    box(menu, 0, 0);
    refresh();
    wrefresh(menu);
    keypad(menu, TRUE);
    
    // declarations
    int choice;
    int highlight = 1;
    int level = 1;
    char choices[][51] = {
        "                                                  ",
        "  N E W  G A M E                                  ",
        "                                                  ",
        "  L e v e l s                                     ",
        "                                                  ",
        "  H e l p                                         ",
        "                                                  ",
        "  Q u i t                                         ",
        "                                                  ",
        "                                                  ",
    };
    
    menu_start:
    while(1){
        
        // handles scrolling through options
        for(int i = 1; i < 10; i += 2){
            if(i == highlight){
                wattron(menu, A_REVERSE);
            }
            mvwprintw(menu, i+1, 1, choices[i]);
            wattroff(menu, A_REVERSE);
        }
        choice = wgetch(menu);
        switch(choice){
            case KEY_UP:
                highlight -= 2;
                if(highlight == -1){
                    highlight = 7;
                }
                break;
            case KEY_DOWN:
                highlight += 2;
                if(highlight == 9){
                    highlight = 1;
                }
                break;
        }
        
        // ends menu window or starts help window, depending on the choice made
        if(choice == 10){
            if(choices[highlight][2] == 'H'){
                mvwprintw(menu, 1, 1, "                                                  ");
                wattron(menu, A_REVERSE);
                mvwprintw(menu, 2, 1, "                 IN-GAME COMMANDS                 ");
                wattroff(menu, A_REVERSE);
                mvwprintw(menu, 3, 1, "                                                  ");
                mvwprintw(menu, 4, 1, "  M  =  m e n u                                   ");
                mvwprintw(menu, 5, 1, "                                                  ");
                mvwprintw(menu, 6, 1, "  P  =  p a u s e                                 ");
                mvwprintw(menu, 7, 1, "                                                  ");
                mvwprintw(menu, 8, 1, "  Q  =  q u i t                                   ");
                wrefresh(menu);
                getchar();
                continue;
            }
            break;
        }
    }

    // if Quit is selected, ends the game
    if(choices[highlight][2] == 'Q'){
        endwin();
        return 0;
    }
    
    // if Levels is selected, displays the levels available to play
    if(choices[highlight][2] == 'L'){
        highlight = 1;
        while(1){
            // handles scrolling through options
            for(int i = 1; i < (LEVELS * 2); i += 2){
                if(i == highlight){
                    level = (i + 1) / 2;
                    wattron(menu, A_REVERSE);
                }
                mvwprintw(menu, i+1, 1, "                   L E V E L  %d                   ", (i+1)/2);
                wattroff(menu, A_REVERSE);
            }
            choice = wgetch(menu);
            switch(choice){
                case KEY_UP:
                    highlight -= 2;
                    if(highlight == -1){
                        highlight = (LEVELS*2-1);
                    }
                    break;
                case KEY_DOWN:
                    highlight += 2;
                    if(highlight == (LEVELS*2+1)){
                        highlight = 1;
                    }
                    break;
                case '\n':
                    break;
                default:
                    goto menu_start;
                    break;
            }
            if(choice == 10){
                break;
            }
        }
    }

    // if New game or Levels is selected, returns level requested, plays the game
    return level;
}

void draw(char prechod_arr[LINES][COLS], int level){
    float prechod1, prechod, offset;
    int offset_counter = 0;
    
    // set speed
    ts.tv_nsec = 0.001 * 10000000L;
    
    // draws the car
    attron(COLOR_PAIR(2));
    mvprintw(LINES/2, 3, " O=o");
    attroff(COLOR_PAIR(2));
        
    // reset the world and draw borders
    for(int x = 0; x < COLS; x++){
        for(int y = 0; y < LINES; y++){
            if(y == 0 || y == LINES-1 || x == 0){
                attron(COLOR_PAIR(1));
                mvprintw(y, x, "#");
                attroff(COLOR_PAIR(1));
                prechod_arr[y][x] = '|';
            } else if(x == COLS-1){
                mvprintw(y, x, "#");
            } else {
                mvprintw(y, x, " ");
                prechod_arr[y][x] = ' ';
            }
        }
    }

    // draw grass :)
    for(int x = 0; x < COLS - 4; x+=6){
        for(int y = 0; y < LINES - 3; y+=3){
            attron(COLOR_PAIR(3));
            mvprintw(y+2, x+2, "\\");
            mvprintw(y+2, x+3, "|");
            mvprintw(y+2, x+4, "/");
            attroff(COLOR_PAIR(3));
        }
    }

    // draw obstacles
    prechod = (LINES-PRECHOD+level)/2;
    for(int x = 0; x < COLS - 2; x++){
        if(x % (30-level*2) == 0 || x == 0){
            offset_counter = 0;
            prechod1 = prechod;
            prechod = rand() % (LINES-PRECHOD-4+level) + 2;
            // v makes sure there is a parking lot at start, otherwise generates next obstacle randomly ^
            if(x / (30-level*2) == 0){
                prechod = (LINES-PRECHOD+level)/2;
            }
            // math, by how many lines does the '=' have to move to get to next obstacle smoothly 
            offset = (prechod1 > prechod) ? (prechod1-prechod)/(30-level*2) : (prechod-prechod1)/(30-level*2);
        }

        // creates road
        for(int y = 0; y < LINES - 2; y++){
            int prechod_upper, prechod_lower;
            if(x % (30 - level*2) == 0){
                // uncomment to see where road changes/ line is
                //mvprintw(y+1, x+1, "|");
                prechod_upper = (int)(prechod1);
                prechod_lower = (int)(prechod1 + PRECHOD - level);
            } else {
                if(prechod1 > prechod){
                    prechod_upper = (int)(prechod1 - offset * offset_counter);
                    prechod_lower = (int)(prechod1 + PRECHOD - level - offset * offset_counter);
                } else {
                    prechod_upper = (int)(prechod1 + offset * offset_counter);
                    prechod_lower = (int)(prechod1 + PRECHOD - level + offset * offset_counter);
                }
            }
            for(int i = (prechod_upper-1); i < (prechod_lower+2); i++){
                mvprintw(i, x+1, " ");
            }
            mvprintw(prechod_upper, x+1, "=");
            mvprintw(prechod_lower, x+1, "=");
            prechod_arr[prechod_upper][x+1] = '|';
            prechod_arr[prechod_lower][x+1] = '|';
            
            refresh();
            nanosleep(&ts, NULL);
        }
        // increases with every column
        offset_counter++;
    }
}

int game(char prechod_arr[LINES][COLS], int level){
    int input, y;
    
    // sets speed for car
    ts.tv_nsec = (SPEED - (0.009 * level)) * 1000000000L;

    // game
    y = LINES/2;
    for(int x = 3; x < COLS - 4; x++){
            
        // check collision with obstacle
        for(int i = 0; i < (LEVELS*2 - level); i++){
            if(prechod_arr[y][x+4] == '|' || prechod_arr[y][x] == '|' || prechod_arr[y][x-1] == '|'){
                mvprintw(LINES/2, COLS/2, "Game Over!");
                attron(COLOR_PAIR(2));
                mvprintw(y, x, " ***");
                attroff(COLOR_PAIR(2));
                refresh();
                sleep(1);
                return 2;
                break;
            }
        }

        // check input
        input = getch();
        switch(input){
            case KEY_DOWN: case 's': case 'S':
                mvprintw(y, x, "    ");
                y++;
                break;
            case KEY_UP: case 'w': case 'W':
                mvprintw(y, x, "    ");
                y--;
                break;
            case 'q': case 'Q':
                mvprintw(LINES/2, COLS/2 - 4, "Quit game");
                refresh();
                sleep(1);
                endwin();
                return 0;
                break;
            case 'p': case 'P':
                mvprintw(LINES/2, COLS/2 - 2, "Pause");
                mvprintw(LINES/2 + 2, COLS/2 - 12, "Press any key to continue");
                refresh();
                getchar();
                getchar();
                mvprintw(LINES/2, COLS/2 - 2, "     ");
                mvprintw(LINES/2 + 2, COLS/2 - 12, "                         ");
                refresh();
                break;
            case 'm': case 'M':
                for(int i = 0; i < LINES; i++){
                    for(int j = 0; j < COLS; j++){
                        mvprintw(i, j, " ");
                    }
                }
                refresh();
                return 2;
                break;
        }
        attron(COLOR_PAIR(2));
        mvprintw(y, x, " O=o");
        attroff(COLOR_PAIR(2));
        refresh();
        nanosleep(&ts, NULL);
    }
    return 1;
}
