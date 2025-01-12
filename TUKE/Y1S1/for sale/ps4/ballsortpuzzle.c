#include <stdio.h>
#include <stdlib.h> 
#include "ballsortpuzzle.h"
#include <time.h>
#include <ctype.h>

//======================================================================================================================
void generator(const int rows, const int columns, char field[rows][columns]){
  const int VERBOSE=1; // print additional diagnostics info
  const char BALLS[] = {'@','#','*','$','%','&','>','~','<','%','^','+','=','?','A','B','O','Q','R','W'};
  // pre-fill whole field with spaces ' '
  for(int row=0;row<rows;row++)
    for(int col=0;col<columns;col++)
      field[row][col] = ' ';
  if(VERBOSE) printf("empty field:\n");
  if(VERBOSE) game_field(rows, columns, field);

  // pre-fill balls' line for next colurs pick-up
  char BALLS_IN_LINE[rows*(columns-2)]; 
  for(int row=0;row<rows;row++)
    for(int col=0;col<columns-2;col++)           // NOTE:2 columns are empty
      BALLS_IN_LINE[col*rows+row] = BALLS[col];

  // mix balls in field
  srand(time(NULL));
  int ball_last_number=(rows*(columns-2)-1); //initial last ball number in BALLS_IN_LINE
  for(int row=0;row<rows;row++)
    for(int col=0;col<(columns-2);col++,ball_last_number--)  // 2 columns are impty
    { int  ball_picked_number = rand() % (ball_last_number+1);
      char ball_picked = BALLS_IN_LINE[ball_picked_number];
      field[row][col] = ball_picked;             // assign picked ball to field
      if(ball_picked_number != ball_last_number) BALLS_IN_LINE[ball_picked_number] = BALLS_IN_LINE[ball_last_number]; // fix empty space
    }
  if(VERBOSE) printf("filled field:\n");
  if(VERBOSE) game_field(rows,columns,field);

  //find columns that will be empty at the initial state  
  int empty_col1 = rand() % columns;
  int empty_col2 = empty_col1;
  //second empty column should be different then first
  while(empty_col2 == empty_col1) empty_col2 = rand() % columns;
  if(VERBOSE) printf("empty_col1: %d\nempty_col2: %d\n",empty_col1+1,empty_col2+1);

  // free empty columns by moving to right side of field
  int column_moved=0;
  if(empty_col1<(columns-2))
  { for(int row=0;row<rows;row++)
    { field[row][columns-2] = field[row][empty_col1];
      field[row][empty_col1] = ' ';    //empty column
    }
    column_moved=1;
    if(VERBOSE) printf("moved/emptied column: %d\n",empty_col1+1);
  }
  if(empty_col2<(columns-2+column_moved))
  { for(int row=0;row<rows;row++)
    { field[row][columns-2+column_moved] = field[row][empty_col2];
      field[row][empty_col2] = ' ';    //empty column
    }
    if(VERBOSE) printf("moved/emptied column: %d\n",empty_col2+1);
  }
  if(VERBOSE) printf("moved columns:\n");
  if(VERBOSE) game_field(rows,columns,field);

  return;
}

//======================================================================================================================
void game_field(const int rows, const int columns, char field[rows][columns]){
  for(int row=0;row<rows;row++)
  { printf(" |");
    for(int col=0;col<columns;col++)
      printf(" %c |",field[row][col]);
    printf("\n");
  }
  printf(" +");
    for(int col=0;col<columns;col++)
      printf("---+");
  printf("\n");
  for(int col=1;col<=columns;col++)
    printf("  %2d",col);
  printf("\n\n");
}

//======================================================================================================================
void down_possible(const int rows, const int columns, char field[rows][columns], int x, int y){
  const int VERBOSE=1; // print additional diagnostics info
  int source_col=x-1;  //internal field value
  int target_col=y-1;  //internal field value
  //checks  
  if((source_col < 0) || source_col > (columns-1))
  {  printf("ERROR: Source column is out of field\n");
     return;
  }
  if((target_col < 0) || target_col > (columns-1))
  {  printf("ERROR: Target column is out of field\n");
     return;
  }
  if(target_col == source_col)
  {  printf("ERROR: same column selected as target\n");
     return;
  }
  if(field[rows-1][source_col] == ' ')
  {  printf("ERROR: Source column is EMPTY/No Balls\n");
     return;
  }
  if(field[0][target_col] != ' ')
  {  printf("ERROR: Target column is FULL\n");
     return;
  }

  //find source ball row
  int source_ball_row = 0;
  for(int row=0; row<rows; row++)
    if(field[row][source_col] != ' ')
    { source_ball_row=row;
      break; //finish for loop at first non-' '
    }
  if(VERBOSE) printf("Source column Ball row is %d:\n", source_ball_row+1);

  //find lower empty row in target column
  int target_empty_row = 0;  //fist is already empty, because of previous check
  for(int row=1;row<rows;row++)
    if(field[row][target_col] == ' ') 
      target_empty_row=row;
    else break; //finish for loop at first non-' '
  if(VERBOSE) printf("Target column empty row is %d:\n", target_empty_row+1);

  //is target column empty ?
  bool target_col_empty=false;
  if(field[rows-1][target_col] == ' ')
    target_col_empty=true;

  //are Balls in Source and Target Columns Compatible ?
  if(! target_col_empty && field[target_empty_row+1][target_col] != field[source_ball_row][source_col])
  {  printf("ERROR: Balls in Source and Target Columns are INCOMPATIBLE...\n");
     return;
  }
  //move Ball and empty it's old position
  field[target_empty_row][target_col] = field[source_ball_row][source_col];
  field[source_ball_row][source_col] = ' ';
  if(VERBOSE) printf("moved %c: from (%d,%d) to (%d,%d)\n",field[target_empty_row][target_col], source_col+1,source_ball_row+1, target_col+1,target_empty_row+1);
}

//======================================================================================================================
bool check(const int rows, const int columns, char field[rows][columns]){
  for(int col=0; col<columns; col++)
  { const char fist_ball=field[0][col];
    for(int row=1; row<rows; row++)
      if(field[row][col] != fist_ball)
        return false; // ball doesn't match with first ball in column
  }
  return true;
}

//======================================================================================================================
void ball_sort_puzzle(){
  const int MAX_BALLS_COUNT=20;
  const int MAX_ROWS=MAX_BALLS_COUNT;
  const int MAX_COLUMNS=MAX_BALLS_COUNT+2;
  printf("\n\n\nHello!\n");
  while(1) //endless while
  { int field_cols=0;
    int field_rows=0;
    printf("\nLets start 'Ball Sort Puzzle' Game!\n");
    while(field_cols<4 || field_cols > MAX_COLUMNS){
      printf("Please, enter Valid Value for Field's WIDTH(4-%d): ",MAX_COLUMNS);
      scanf("%d", &field_cols);
    }
    while(field_rows<2 || field_rows > MAX_ROWS){
      printf("Please, enter Valid Value for Field's HEIGHT(2-%d): ",MAX_ROWS);
      scanf("%d", &field_rows);
    }
    //declare Field array
    char field[field_rows][field_cols];
    //fill the Field
    generator(field_rows,  field_cols, field);
    game_field(field_rows, field_cols, field);
    if(check(field_rows, field_cols, field)){
      printf("It is AMAZING!!!\n");
      printf("You are so LUCKY, so your Field was generated as ALREADY SORTED!\n");
      printf("Have You ever been in Las Vegas ?...\n");
    }
    else{
      //endless look until Field in the Right State
      while(!check(field_rows, field_cols, field)){
        //input move details
        int source_col;
        printf("Source column: ");
        scanf("%d", &source_col);
        int target_col;
        printf("Target column: ");
        scanf("%d", &target_col);
        printf("\n");

        //do move
        down_possible(field_rows, field_cols, field, source_col, target_col);
        //show Field
        game_field(field_rows, field_cols, field);
      }
      printf("Congratulations! You won!\n\n");
    }
    char continue_answer = ' ';
    do {
      int cc; while((cc = getchar()) != '\n');
      printf("\nWould you like to play once more ?[Y/n]: ");
      scanf("%c", &continue_answer);
      if(continue_answer == '\n') continue_answer='Y';
      continue_answer=toupper(continue_answer);
    } while (continue_answer != 'Y' && continue_answer != 'N');
    if(continue_answer=='N')
    { printf("Thank you for your time!\nSee you later.");
      return;
    }
  } //endless while
}

