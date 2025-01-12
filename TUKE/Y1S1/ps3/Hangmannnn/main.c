#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "hangman.h"

int main(){
    srand(time(NULL));
    char secret[50];
    get_word(secret);
    hangman(secret);
    
    return 0;
}
