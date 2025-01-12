#include <stdio.h>
#include <stdlib.h>

#define TARGET_WORD_LOWER_B "bananas"
#define TARGET_WORD_UPPER_B "BANANAS"
#define TARGET_WORD_LOWER_A "ananas"
#define TARGET_WORD_UPPER_A "ANANAS"
#define TARGET_LENGTH 6

#include <stdio.h>

void writeCountToFile(FILE *file, int count) {
    if (count < 10) {
        fputc(count + '0', file);
    } else {
        int digits[10];
        int temp = count;
        int digit, i = 0;

        while (temp > 0) {
            digit = temp % 10;
            digits[i++] = digit;
            temp /= 10;
        }
        
        for (i = i - 1; i >= 0; i--) {
            fputc(digits[i] + '0', file);
        }
    }
}



int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <filename>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        printf("Incorrect filename\n");
        return 1;
    }

    int count = 0;
    int symbol = 0;
    char element;
    while ((element = fgetc(file)) != EOF) {
        if (element == TARGET_WORD_LOWER_A[symbol] || element == TARGET_WORD_UPPER_A[symbol] || element == TARGET_WORD_LOWER_B[symbol] || element == TARGET_WORD_UPPER_B[symbol]) {
            symbol++;
        } else {
            symbol = 0;
        }
        if (symbol == TARGET_LENGTH) {
            count++;
            symbol = 0;
        }
    }
    fclose(file);

    file = fopen(argv[1], "w");
    writeCountToFile(file, count);
    fclose(file);

    return 0;
}
