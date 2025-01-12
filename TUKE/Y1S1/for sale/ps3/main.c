#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "hangman.h"
#include <stdbool.h>
#include <ctype.h>
//#include "morse.h"

int main()
{
    char secret[] = "";
    get_word(secret);
    hangman(secret);

    return 0;
}
