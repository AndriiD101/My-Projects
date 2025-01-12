#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include "bmp.h"

char* reverse(const char* text) {
    if(text == NULL) return NULL;
    
    int length = strlen(text);
    char* reversed = malloc(length + 1);
    char *text_copy = malloc(length + 1);
    text_copy = strcpy(text_copy, text);
    for(int i = 0; i < length; i++){
        text_copy[i] = toupper(text_copy[i]);
    }

    for(int i = 0; i < length; i++) {
        reversed[i] = text_copy[length - i - 1];
    }

    reversed[length] = '\0';
    free(text_copy);
    return reversed;
}



char* vigenere_encrypt(const char* key, const char* text) {
    if (!key || !text) return NULL;
    for (int i = 0; key[i] != '\0'; i++) {
        if (!isalpha(key[i]))
            return NULL;
    }

    // for (int i = 0; text[i] != '\0'; i++) {
    //     if (!isalpha(text[i]))
    //         return NULL;
    // }

    int key_len = strlen(key);
    int text_len = strlen(text);
    char* encrypted_text = (char*)malloc((text_len + 1) * sizeof(char));
    if (encrypted_text == NULL) {
        return NULL; 
    }

    for (int i = 0, j = 0; i < text_len; i++) {
        char current_char = text[i];
        if (isalpha(current_char)) {
            char key_char = toupper(key[j % key_len]);
            int shift = key_char - 'A';
            if (islower(current_char))
                encrypted_text[i] = ((current_char - 'a' + shift) % 26) + 'A';
            else
                encrypted_text[i] = ((current_char - 'A' + shift) % 26) + 'A';
            j++;
        } else {
            encrypted_text[i] = current_char;
        }
    }
    encrypted_text[text_len] = '\0';
    return encrypted_text;
}

char* vigenere_decrypt(const char* key, const char* text) {
    if (!key || !text) return NULL;
    for (int i = 0; key[i] != '\0'; i++) {
        if (!isalpha(key[i]))
            return NULL;
    }

    // for (int i = 0; text[i] != '\0'; i++) {
    //     if (!isalpha(text[i]))
    //         return NULL;
    // }
    int key_len = strlen(key);
    int text_len = strlen(text);
    char* decrypted_text = (char*)malloc((text_len + 1) * sizeof(char));
    if (decrypted_text == NULL) {
        return NULL; // Unable to allocate memory
    }

    for (int i = 0, j = 0; i < text_len; i++) {
        char current_char = text[i];
        if (isalpha(current_char)) {
            char key_char = toupper(key[j % key_len]);
            int shift = key_char - 'A';
            if (islower(current_char))
                decrypted_text[i] = ((current_char - 'a' - shift + 26) % 26) + 'A';
            else
                decrypted_text[i] = ((current_char - 'A' - shift + 26) % 26) + 'A';
            j++;
        } else {
            decrypted_text[i] = current_char;
        }
    }
    decrypted_text[text_len] = '\0';
    return decrypted_text;
}

void encode_string(const char string[], bool bytes[strlen(string)+1][8])
{
    char string_element;
    int letter_ascii;
    for(int elm = 0; elm < strlen(string); elm++)
    {
        string_element = string[elm];
        letter_ascii = string_element;
        for(int div = 7; div >= 0; div--)
        {
        if(letter_ascii%2 == 0){
            bytes[elm][div] = letter_ascii%2;
            letter_ascii/=2;
        }
        else {
            bytes[elm][div] = letter_ascii%2;
            letter_ascii/=2;
        }
        }
    }
    for(int i = 0; i < 8; i++)
    {
        bytes[strlen(string)][i] = 0;
    }
}

double power(double base, int exponent) {
    double result = 1.0;
    int i;

    if (exponent < 0) {
        base = 1.0 / base;
        exponent = -exponent;
    }

    for (i = 0; i < exponent; i++) {
        result *= base;
    }

    return result;
}


// unsigned char* bit_encrypt(const char* text) {
//     if (!text) return NULL;

//     size_t len = strlen(text);
//     unsigned char* encrypted = (unsigned char*)malloc(len + 1);
//     char* text_copy = malloc(len + 1);
//     text_copy = strcpy(text_copy, text);
//     for(int i = 0; i < len; i++){
//         text_copy[i] = toupper((unsigned char)text_copy[i]);
//     }

//     if (!encrypted) return NULL;

//     for (size_t index = 0; index < len; index++) {
//         int byte[8];
//         int num = text_copy[index];

//         // Decimal to binary
//         for (int i = 7; i >= 0; i--) {
//             byte[i] = num % 2;
//             num /= 2;
//         }

//         // Reverse swap
//         bool temp = byte[1];
//         byte[1] = byte[0];
//         byte[0] = temp;
//         temp = byte[3];
//         byte[3] = byte[2];
//         byte[2] = temp;

//         // Reverse XOR operation
//         for (int i = 0; i < 4; i++) {
//             byte[i + 4] = byte[i] ^ byte[i + 4];
//         }

//         // Binary to decimal
//         int decimal = 0;
//         for (int i = 0; i < 8; i++) {
//             decimal += byte[i] * (1 << (7 - i));
//         }
//         encrypted[index] = (unsigned char)decimal;
//     }
//     encrypted[len] = '\0';
//     free(text_copy);
//     return encrypted;
// }


unsigned char* bit_encrypt(const char* text) {
    if (!text) return NULL;
    size_t len = strlen(text);
    bool bytes[len+1][8];
    unsigned char* encrypted = (unsigned char*)malloc(len + 1);
    encode_string(text, bytes);

    // Encryption steps
	// swap
    for(int rows = 0; rows < len; rows++) {
        bool temp = bytes[rows][0];
        bytes[rows][0] = bytes[rows][1];
        bytes[rows][1] = temp;
        temp = bytes[rows][2];
        bytes[rows][2] = bytes[rows][3];
        bytes[rows][3] = temp;
    }

    //XOR
    for(int rows = 0; rows < len; rows++) {
        for(int i = 4; i < 8; i++) {
            bytes[rows][i] = bytes[rows][i] ^ bytes[rows][i - 4];
        }
    }

    // Convert binary to decimal
    for (int row = 0; row < len; row++) {
        unsigned char decimal = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (bytes[row][bit] == 1) {
                decimal += power(2, 7 - bit);
            }
        }
        encrypted[row] = decimal;
    }

    encrypted[len] = '\0';
    return encrypted; 
}

char* bit_decrypt(const unsigned char* text) {
    if (!text) return NULL;

    size_t len = strlen((const char*)text);
    char* decrypted = (char*)malloc(len + 1); 

    if (!decrypted) return NULL;

    for (size_t index = 0; index < len; index++) {
        int byte[8];
        int num = text[index];

        // Decimal to binary
        for (int i = 7; i >= 0; i--) {
            byte[i] = num % 2;
            num /= 2;
        }

        // XOR operation
        for (int i = 0; i < 4; i++) {
            byte[i + 4] = byte[i] ^ byte[i + 4];
        }

        // Swap
        bool temp = byte[0];
        byte[0] = byte[1];
        byte[1] = temp;
        temp = byte[2];
        byte[2] = byte[3];
        byte[3] = temp;

        // Binary to decimal
        int decimal = 0;
        for (int i = 0; i < 8; i++) {
            decimal += byte[i] * power(2, 7 - i);
        }
        decrypted[index] = (char)decimal;
    }
    decrypted[len] = '\0'; 
    return decrypted;
}

unsigned char* bmp_encrypt(const char* key, const char* text) {
    if (!key || !text) return NULL;
    for (int i = 0; key[i] != '\0'; i++) {
        if (!isalpha(key[i]))
            return NULL;
    }
    
    char* rev_str = reverse(text);
    char* vig_str_enc = vigenere_encrypt(key, rev_str);
    unsigned char* bit_str_enc = bit_encrypt(vig_str_enc);
    free(rev_str);
    free(vig_str_enc);
    return bit_str_enc;
}

char* bmp_decrypt(const char* key, const unsigned char* text) {
    if (!key || !text) return NULL;
    for (int i = 0; key[i] != '\0'; i++) {
        if (!isalpha(key[i]))
            return NULL;
    }

    char* bit_str_enc = bit_decrypt(text);
    char* vig_str_enc = vigenere_decrypt(key, bit_str_enc);
    char* rev_str = reverse(vig_str_enc);
    free(bit_str_enc);
    free(vig_str_enc);
    return rev_str;
}

// int main() {
//     unsigned char* encrypted;

//     // basic test with long text
//     char word[] = "Jaj beda mne skrikla matka uz je tam po mojej krave";
//     int len = strlen(word);
//     encrypted = bit_encrypt(word);
//     for(int i=0; encrypted[i]!='\0';i++) {
//         printf("%x ", encrypted[i]);
//     }
//     printf("\n");

//     // Decrypt
//     char* decrypted = bit_decrypt(encrypted);
//     printf("Decrypted: %s\n", decrypted);

//     // const char* key = "KEY";

//     // char* encrypted_text = vigenere_encrypt(key, decrypted);
//     // if (encrypted_text == NULL) {
//     //     printf("Encryption failed. Invalid input.\n");
//     //     return 1;
//     // }
//     // printf("Encrypted text: %s\n", encrypted_text);

//     // free(encrypted_text); // Don't forget to free allocated memory
//     free(encrypted);
//     // free(decrypted);
//     return 0;


//     return 0;
// }
