#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "morse.h"
#include <ctype.h>
#include <stdbool.h>

#define MAX_MORSE_LENGTH 100

void text_to_morse(const char text[], char output[]) {
    char copy[100];
    strcpy(copy, text);
    char alphabet[][10] = {".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---",
                    "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-",
                    "...-", ".--", "-..-", "-.--", "--.."};
    char nums_morse[][10] = {"-----", ".----", "..---", "...--", "....-", ".....", "-....", "--...", "---..", "----."};
    int i = 0;
    char buffer[100];
    output[0] = '\0';

    while (copy[i] != '\0') {
        if (isalpha(copy[i])) {
            strcpy(buffer, alphabet[toupper(copy[i]) - 65]);
        } else if (isdigit(copy[i])) {
            strcpy(buffer, nums_morse[copy[i] - 48]);
        } else {
            strcpy(buffer, "/");
        }
        strcat(output, buffer);
        if (copy[i + 1] != '\0') {
            strcat(output, " ");
        }
        i++;
    }
    output[strlen(output)] = '\0';
}

void morse_to_text(const char morse[], char output[]) {
    typedef struct {
    char morseCode[9];
    char letter;
    } MorseCodeEntry;
    MorseCodeEntry morseCodeTable[] = {
    {".-", 'A'},
    {"-...", 'B'},
    {"-.-.", 'C'},
    {"-..", 'D'},
    {".", 'E'},
    {"..-.", 'F'},
    {"--.", 'G'},
    {"....", 'H'},
    {"..", 'I'},
    {".---", 'J'},
    {"-.-", 'K'},
    {".-..", 'L'},
    {"--", 'M'},
    {"-.", 'N'},
    {"---", 'O'},
    {".--.", 'P'},
    {"--.-", 'Q'},
    {".-.", 'R'},
    {"...", 'S'},
    {"-", 'T'},
    {"..-", 'U'},
    {"...-", 'V'},
    {".--", 'W'},
    {"-..-", 'X'},
    {"-.--", 'Y'},
    {"--..", 'Z'},
    {".----", '1'},
    {"..---", '2'},
    {"...--", '3'},
    {"....-", '4'},
    {".....", '5'},
    {"-....", '6'},
    {"--...", '7'},
    {"---..", '8'},
    {"----.", '9'},
    {"-----", '0'},
};

    char morseCopy[strlen(morse) + 1];
    strcpy(morseCopy, morse);
    int numMorseCodeEntries = sizeof(morseCodeTable) / sizeof(MorseCodeEntry);
    int outputIndex = 0;

    for (char *token = strtok(morseCopy, " "); token != NULL; token = strtok(NULL, " ")) {
        bool found = false;
        for (int i = 0; i < numMorseCodeEntries; i++) {
            char *morseSymbol = morseCodeTable[i].morseCode;
            char letter = morseCodeTable[i].letter;

            if (strcmp(token, morseSymbol) == 0) {
                output[outputIndex++] = letter;
                found = true;
                break;
            }
        }

        if (!found) {
            output[outputIndex++] = '?';
        }
    }

    output[outputIndex] = '\0';
}

int is_morse_code_valid(const char morse[]) {
    char morse_code_copy[MAX_MORSE_LENGTH];
    strncpy(morse_code_copy, morse, sizeof(morse_code_copy) - 1);
    morse_code_copy[sizeof(morse_code_copy) - 1] = '\0';
    char *token = strtok(morse_code_copy, " "); 
    char *morse_code_dict[] = {
        ".-", "-...", "-.-.", "-..", ".", "..-.", "--.", "....", "..", ".---", "-.-", ".-..", "--", "-.", "---", ".--.", "--.-", ".-.", "...", "-", "..-", "...-", ".--", "-..-", "-.--", "--..", ".----", "..---", "...--", "....-", ".....", "-....", "--...", "---..", "----.", "-----", "/"};

    while (token != NULL) {
        int is_valid = 0;
        for (int i = 0; i < sizeof(morse_code_dict) / sizeof(morse_code_dict[0]); ++i) {
            if (strcmp(token, morse_code_dict[i]) == 0) {
                is_valid = 1;
                break;
            }
        }

        if (!is_valid) {
            return 0;
        }

        token = strtok(NULL, " ");
    }

    return 1;
}


/*int main()
{
    char output[150];

    text_to_morse("HELLO", output);
    puts(output);
    //prints: .... . .-.. .-.. ---
    morse_to_text(".... . .-.. .-.. ---", output);
    //prints: HELLO
    if (is_morse_code_valid(".... . .-.. .-.. ---")) {
        printf("Code is valid! \n");
    } else {
        printf("Code is invalid! \n");
    }
}*/
