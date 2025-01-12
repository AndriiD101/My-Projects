#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <string.h>
#include <ctype.h>
#include "hangman.h"

int get_word(char secret[]){
    // check if file exists first and is readable
    FILE *fp = fopen(WORDLIST_FILENAME, "rb");
    if( fp == NULL ){
        fprintf(stderr, "No such file or directory: %s\n", WORDLIST_FILENAME);
        return 1;
    }

    // get the filesize first
    struct stat st;
    stat(WORDLIST_FILENAME, &st);
    long int size = st.st_size;

    do{
        // generate random number between 0 and filesize
        long int random = (rand() % size) + 1;
        // seek to the random position of file
        fseek(fp, random, SEEK_SET);
        // get next word in row ;)
        int result = fscanf(fp, "%*s %20s", secret);
        if( result != EOF )
            break;
    }while(1);

    fclose(fp);

    return 0;
}

int is_word_guessed(const char secret[], const char letters_guessed[]){
    int secretlen = strlen(secret);
    int letterslen = strlen(letters_guessed);
    int result = 0;

    for(int i = 0; i < secretlen; i++){
        for(int j = 0; j < letterslen; j++){
            if(secret[i] == letters_guessed[j]){
                result += 1;
                break;
            }
        }
    }
    if(result >= secretlen){
        return 1;
    }
    return 0;
}

void get_guessed_word(const char secret[], const char letters_guessed[], char guessed_word[]){
    int secretlen = strlen(secret);
    int letterslen = strlen(letters_guessed);

    for(int i = 0; i < secretlen; i++){
        guessed_word[i] = '_';
    }
    for(int i = 0; i < secretlen; i++){
        for(int j = 0; j < letterslen; j++){
            if(secret[i] == letters_guessed[j]){
                guessed_word[i] = secret[i];
            }
        }
    }
    guessed_word[secretlen] = '\0';
}

void get_available_letters(const char letters_guessed[], char available_letters[]){
    char alphabet[] = "abcdefghijklmnopqrstuvwxyz";
    int guessedlen = strlen(letters_guessed);
    int alplen = strlen(alphabet);

    for(int i = 0; i < alplen; i++){
        available_letters[i] = alphabet[i];
    }
    available_letters[alplen] = '\0';
    for(int i = 0; i < guessedlen; i++){
        for(int j = 0; j < alplen; j++){
            if(letters_guessed[i] == available_letters[j]){
                for(int k = j; k < alplen; k++){
                    available_letters[k] = available_letters[k + 1];
                }
                break;
            }
        }
    }
    int avallen = strlen(available_letters);
    available_letters[avallen] = '\0';
}

void clear (void) {
    while (getchar() != '\n');
}

void hangman(const char secret[]){
    int secretlen = strlen(secret); 
    int avallen, successrate, oops_same_letter = 0; 
    int g = 8;
    char aval_letters[50], guess_letters[50], word[50], word1[50];

    printf("Welcome to the game, Hangman!\n");
    printf("I am thinking of a word that is %d letters long.\n", secretlen);

    for(int i = 0; i < 100; i++){
        successrate = 0;
        oops_same_letter = 0;
        printf("-------------\n");
        printf("You have %d guesses left.\n", g);
        get_available_letters(guess_letters, aval_letters);
        avallen = strlen(aval_letters);
        printf("Available letters: %s\n", aval_letters);
        printf("Please guess a letter: ");
        scanf("%s", &guess_letters[i]);
        if(isupper(guess_letters[i])){
            guess_letters[i] = tolower(guess_letters[i]);
        }
        get_guessed_word(secret, guess_letters, word1);
        clear();
        for(int j = 0; j < secretlen; j++){
            word[j * 2] = word1[j];
            word[(j * 2) + 1] = ' ';
        }
        word[secretlen * 2-1] = '\0';
        if(guess_letters[i + 1] > 'a' && guess_letters[i + 1] < 'z'){
            if(is_word_guessed(secret, guess_letters)){
                printf("Congratulations, you won!\n");
            } else {
                printf("Sorry, bad guess. The word was %s.\n", secret);
            }
            break;
        } 
        if(guess_letters[i] < 'a' || guess_letters[i] > 'z'){
            printf("Oops! '%c' is not a valid letter: %s\n", guess_letters[i], word);
            continue;
        }
        for(int j = 0; j < avallen; j++){
            if(guess_letters[i] != aval_letters[j]){
                oops_same_letter += 1;
            }
        }
        if(oops_same_letter == avallen){
            printf("Oops! You've already guessed that letter: %s\n", word);
            continue;
        }
        for(int j = 0; j < secretlen; j++){
            if(guess_letters[i] == secret[j]){
                successrate += 1;
            }
        }
        if(successrate > 0){
            printf("Good guess: %s\n", word);
            if(is_word_guessed(secret, guess_letters)){
                printf("-------------\n");
                printf("Congratulations, you won!\n");
                break;
            }    
        } else {
            printf("Oops! That letter is not in my word: %s\n", word);
            g--;
        }
        if(g == 0){
            if(is_word_guessed(secret, guess_letters)){
                printf("-------------\n");
                printf("Congratulations, you won!\n");
            } else {
                printf("-------------\n");
                printf("Sorry, you ran out of guesses. The word was %s.\n", secret);
            }
            break;
        }
    }
}
