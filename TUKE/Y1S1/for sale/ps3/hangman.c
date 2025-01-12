#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "hangman.h"
#include <stdbool.h>
#include <ctype.h>

//abcdefghijklmnopqrstuvwxyz

int is_word_guessed(const char secret[], const char letters_guessed[])
{
    int size_secret = strlen(secret);
    int size_letters_guessed = strlen(letters_guessed);
    int flag = 0;
    for( int i = 0; i < size_secret; i++ )
    {
        for ( int j = 0; j < size_letters_guessed; j++)
        {
            if( secret[i] == letters_guessed[j])
            {
                flag += 1;
                break;
            }
        }
    }

    if(flag >= size_secret)
    {
        return 1;
    }
    return 0;
}

void get_guessed_word(const char secret[], const char letters_guessed[], char guessed_word[])
{
    int len = strlen(secret);
    int len_letters = strlen(letters_guessed);
    for (int i = 0; i < len; i++)
    {
        guessed_word[i] = '_';
    }
    for(int i = 0; i < len; i++)
    {
        for(int j = 0; j < len_letters; j++)
        {
            if(secret[i] == letters_guessed[j])
            {
                guessed_word[i] = letters_guessed[j];
            }
        }
    }
    guessed_word[len] = '\0';
}

void get_available_letters(const char letters_guessed[], char available_letters[]) {
  strcpy(available_letters, "abcdefghijklmnopqrstuvwxyz");
  for (int i = 0; letters_guessed[i] != '\0'; i++) {
    for (int j = 0; available_letters[j] != '\0'; j++) {
      if (letters_guessed[i] == available_letters[j]) {
        available_letters[j] = '_';
        break;
      }
    }
  }
  int available_index = 0;
  for (int i = 0; available_letters[i] != '\0'; i++) {
    if (available_letters[i] != '_') {
      available_letters[available_index++] = available_letters[i];
    }
  }
  available_letters[available_index] = '\0';
}

void hangman(const char secret[]) 
{
    int len_of_word = strlen(secret);
    int len_available, success, same;
    int lives = 8;
    char available_letters[50], guessed[50], word[50], word1[50];
    int i = 0;

  printf("Welcome to the game, Hangman!\n");
  printf("I am thinking of a word that is %d letters long.\n", len_of_word);
    while (1) 
    {
    same = 0;
    success = 0;
    printf("-------------\n");
    printf("You have %d guesses left.\n", lives);
    get_available_letters(guessed, available_letters);
    len_available = strlen(available_letters);
    printf("Available letters: %s\n", available_letters);
    printf("Please guess a letter: ");
    scanf("%s", &guessed[i]);
    if (isupper(guessed[i])) {
        guessed[i] = tolower(guessed[i]);
    }
    get_guessed_word(secret, guessed, word1);
    for (int j = 0; j < len_of_word; j++) {
        word[j * 2] = word1[j];
        word[(j * 2) + 1] = ' ';
    }
    word[len_of_word * 2 - 1] = '\0';
    if(guessed[i + 1] > 'a' && guessed[i + 1] < 'z'){
            if(is_word_guessed(secret, guessed)){
                printf("Congratulations, you won!\n");
            } else {
                printf("Sorry, bad guess. The word was %s.\n", secret);
            }
            break;
    }
    if (guessed[i] < 'a' || guessed[i] > 'z') {
        printf("Oops! '%c' is not a valid letter: %s\n", guessed[i], word);
        continue;
    }
    for (int j = 0; j < len_available; j++) {
        if (guessed[i] != available_letters[j]) {
        same++;
        }
    }
    if (same == len_available) {
        printf("Oops! You've already guessed that letter: %s\n", word);
        continue;
    }
    for (int j = 0; j < len_of_word; j++) {
        if (guessed[i] == secret[j]) {
        success++;
        }
    }
    if (success > 0) {
        printf("Good guess: %s\n", word);
        if(is_word_guessed(secret, guessed)){
                printf("-------------\n");
                printf("Congratulations, you won!\n");
                break;
            }
    } else {
        printf("Oops! That letter is not in my word: %s\n", word);
        lives--;
    }
    if (lives == 0) {
        if(is_word_guessed(secret, guessed)){
                printf("-------------\n");
                printf("Congratulations, you won!\n");
            } else {
                printf("-------------\n");
                printf("Sorry, you ran out of guesses. The word was %s.\n", secret);
            }
        break;
    }
    i++;
    }
}

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

//int main()
//{
//    hangman("squawcked");
//    return 0;
//}
