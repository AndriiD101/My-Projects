#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

#define MAX_ROW 5
#define MAX_COL 5
#define ALPHA "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

void find_coordinates_of_the_letter(char **arr, char ch, int *x, int *y);
void swap(int *a, int *b);
char toupper_case(char c);
char* put_x_change_W_to_V_toupper_case(const char* text);
char** Create_key_word_array(const char* key);
char* remove_char_index(char* text, int index);
char* clean_to_upper_W_to_V(const char* text); 
char* remove_duplicate(const char* text);
void free_memory_of_all_array(char **arr);
char* rewrite_cipher_with_matrix_key(char** key, char* text, int orientation);
char* paste_character_and_index(char* arr, int position, char ch);
char* cal_set_diff(char* set_A, char* set_B); // A \ B in other words A - B

int main()
{
    char *encrypted, *decrypted;

    encrypted = playfair_encrypt("SeCReT", "Hello world");
    printf("%s", encrypted);
    printf("\n");
    decrypted = playfair_decrypt("SeCReT", encrypted);
    printf("%s", decrypted);
    free(encrypted);
    free(decrypted);
    printf("\n");

    encrypted = playfair_encrypt("world", "Hello");
    printf("%s", encrypted);
    printf("\n");
    decrypted = playfair_decrypt("world", encrypted);
    printf("%s", decrypted);
    free(encrypted);
    free(decrypted);
    printf("\n");

    encrypted = playfair_encrypt("Password", "Taxi please");
    printf("%s", encrypted);
    printf("\n");
    decrypted = playfair_decrypt("Password", encrypted);
    printf("%s", decrypted);
    free(encrypted);
    free(decrypted);
    printf("\n");

    encrypted = playfair_encrypt("please", "Taxxxiii");
    printf("%s", encrypted);
    printf("\n");
    decrypted = playfair_decrypt("please", encrypted);
    printf("%s", decrypted);
    free(encrypted);
    free(decrypted);
    printf("\n");
    return 0;
}

char* rewrite_cipher_with_matrix_key(char** key, char* text, int orientation)
{
    if (key && text)
    {
        int tmp_orientation = orientation >= 0 ? 1 : 4;
        int row1, column1, row2, column2;
        size_t text_len = strlen(text);
        if (!(text_len % 2))
        {
            char *encrypt_text = (char*)calloc((text_len + 1), sizeof(char));
            for (int i = 1; i < text_len; i += 2)
            {
                find_coordinates_of_the_letter(key, text[i - 1], &column1, &row1);
                find_coordinates_of_the_letter(key, text[i], &column2, &row2);
                if (column1 == column2)
                {
                    row1 = (row1 + tmp_orientation) % MAX_ROW;
                    row2 = (row2 + tmp_orientation) % MAX_ROW;
                }
                else if (row1 == row2)
                {
                    column1 = (column1 + tmp_orientation) % MAX_COL;
                    column2 = (column2 + tmp_orientation) % MAX_COL;
                }
                else
                {
                    int temp = column1;
                    column1 = column2;
                    column2 = temp;
                }
                encrypt_text[i - 1] = key[row1][column1];
                encrypt_text[i] = key[row2][column2];
            }
            encrypt_text[text_len] = '\0';
            free(text);
            return encrypt_text;
        }
        free(text);
    }
    return NULL;
}

void swap(int *a, int *b)
{
    int tmp = *a;
    *a = *b;
    *b = tmp;
}

char toupper_case(char c) {
    if (c >= 'a' && c <= 'z') {
        return c - ('a' - 'A');
    } else {
        return c;
    }
}

void find_coordinates_of_the_letter(char **arr, char ch, int *x, int *y)
{
    if (!arr)
    {
        *x = -1;
        *y = -1;
        return;
    }

    char seek_char = toupper_case(ch);
    for (int row = 0; row < 5; row++)
    {
        for (int col = 0; col < 5; col++)
        {
            if (toupper_case(arr[row][col]) == seek_char)
            {
                *y = row;
                *x = col;
                return;
            }
        }
    }

    *x = -1;
    *y = -1;
}

char* remove_char_index(char* text, int index)
{
    if (!text || index < 0)
        return NULL;

    size_t text_len = strlen(text);

    if (index >= text_len)
        return text;

    char *tmp_text = (char*)calloc(text_len, sizeof(char));
    if (!tmp_text)
        return NULL;

    strncpy(tmp_text, text, index);
    strcpy(tmp_text + index, text + index + 1);

    free(text);

    return tmp_text;
}

char* put_x_change_W_to_V_toupper_case(const char* text)
{
    if (!text)
        return NULL;

    char *tmp_text = clean_to_upper_W_to_V(text);
    if (!tmp_text)
        return NULL;

    size_t text_len = strlen(tmp_text);
    for (int i = 1; i < text_len; i += 2)
    {
        if (tmp_text[i - 1] == tmp_text[i] && tmp_text[i] != 'X')
        {
            tmp_text = paste_character_and_index(tmp_text, i, 'X');
            if (!tmp_text) 
                return NULL;
            text_len++;
        }
    }

    if (text_len % 2 != 0)
    {
        tmp_text = paste_character_and_index(tmp_text, text_len, 'X');
        if (!tmp_text)
            return NULL;
    }

    return tmp_text;
}

char* clean_to_upper_W_to_V(const char* text)
{
    if (!text)
        return NULL;

    size_t text_len = strlen(text);
    char *tmp_text = (char*)calloc(text_len + 1, sizeof(char));
    if (!tmp_text)
        return NULL;

    int j = 0;
    for (int i = 0; i < text_len; i++)
    {
        if (isalpha(text[i]))
        {
            tmp_text[j++] = toupper_case(text[i]);
            if (tmp_text[j - 1] == 'W')
                tmp_text[j - 1] = 'V';
        }
    }
    tmp_text[j] = '\0';
    return tmp_text;
}

char* clean_key(char *key)
{
    if (!key)
        return NULL;

    size_t len_key = strlen(key);
    char *new_key = (char*)calloc(len_key + 1, sizeof(char));
    if (!new_key)
        return NULL;

    int j = 0;
    for (int i = 0; i < len_key; i++)
    {
        char c = toupper_case(key[i]);
        if (isalpha(c) || c == ' ')
        {
            if (c == 'W')
                c = 'V';
            new_key[j++] = c;
        }
        else
        {
            free(key);
            free(new_key);
            return NULL;
        }
    }
    new_key[j] = '\0';

    int k = 0;
    for (int i = 0; i < j; i++)
    {
        if (new_key[i] != ' ')
            new_key[k++] = new_key[i];
    }
    new_key[k] = '\0';

    free(key);
    return new_key;
}

char* paste_character_and_index(char* arr, int position, char ch) {
    if (arr && position <= strlen(arr)) {
        arr = realloc(arr, strlen(arr) + 2);
        if (!arr) return NULL; // Check if reallocation failed
        memmove(&arr[position + 1], &arr[position], strlen(arr) - position + 1);
        arr[position] = ch;
    }
    return arr;
}

char* cal_set_diff(char* set_A, char* set_B) {
    int hash[256] = {0};
    size_t len_A = strlen(set_A);
    char* C = (char*)calloc(len_A + 1, sizeof(char));
    int t = 0;

    while (*set_B) {
        hash[toupper(*set_B++)]++;
    }

    while (*set_A) {
        if (hash[toupper(*set_A)] == 0) {
            C[t++] = *set_A;
        }
        set_A++;
    }

    C[t] = '\0';
    return C;
}

char** Create_key_word_array(const char* key)
{
    if (!key)
        return NULL;

    char *short_key = remove_duplicate(key);
    if (!short_key || !*short_key)
        return NULL;

    short_key = clean_key(short_key);
    if (!short_key || !*short_key)
        return NULL;

    char *alpha = cal_set_diff(ALPHA, short_key);
    if (!alpha || !*alpha)
    {
        free(short_key);
        return NULL;
    }

    size_t len_short_key = strlen(short_key);
    char **arr_key = (char**)calloc(5, sizeof(char*));
    if (!arr_key)
    {
        free(short_key);
        free(alpha);
        return NULL;
    }

    for (int i = 0; i < 5; i++)
    {
        arr_key[i] = (char*)calloc(5, sizeof(char));
        if (!arr_key[i])
        {
            free_memory_of_all_array(arr_key);
            free(short_key);
            free(alpha);
            return NULL;
        }
    }

    int k = 0;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            if (k < len_short_key)
            {
                if (isalpha(short_key[k]))
                {
                    arr_key[i][j] = toupper_case(short_key[k++]);
                }
                else
                {
                    free_memory_of_all_array(arr_key);
                    free(short_key);
                    free(alpha);
                    return NULL;
                }
            }
            else
            {
                arr_key[i][j] = toupper_case(alpha[k++ - len_short_key]);
            }
        }
    }

    free(short_key);
    free(alpha);
    return arr_key;
}

char* remove_duplicate(const char* text)
{
    if (text)
    {
        size_t text_len = strlen(text);
        char *tmp_text = (char*)calloc(text_len + 1, sizeof(char));
        size_t len_tmp_text = 0;

        for (size_t i = 0; i < text_len; i++)
        {
            char current_char = toupper_case(text[i]);
            if (current_char == 'W')
            {
                current_char = 'V';
            }

            int found = 0;
            for (size_t j = 0; j < len_tmp_text; j++)
            {
                if (toupper_case(tmp_text[j]) == current_char)
                {
                    found = 1;
                    break;
                }
            }

            if (!found)
            {
                tmp_text[len_tmp_text++] = current_char;
            }
        }
        tmp_text[len_tmp_text] = '\0'; 
        return tmp_text;
    }
    return NULL;
}

void free_memory_of_all_array(char **arr)
{
    if (arr)
    {
        for (size_t i = 0; arr[i] != NULL; i++)
        {
            free(arr[i]);
        }
        free(arr);
    }
}

char* playfair_encrypt(const char* key, const char* text)
{
    if (!key || !text || strlen(text) == 0) return NULL;

    // Check if the key contains only alphabetic characters
    for (int i = 0; key[i] != '\0'; i++) {
        if (!isalpha(key[i]))
            return NULL;
    }

    char **arr_key = Create_key_word_array(key);
    if (!arr_key)
        return NULL;

    char *processed_text = put_x_change_W_to_V_toupper_case(text);
    if (!processed_text)
    {
        free_memory_of_all_array(arr_key);
        return NULL;
    }

    char *encrypted_text = rewrite_cipher_with_matrix_key(arr_key, processed_text, 1);
    free(processed_text);
    if (!encrypted_text)
    {
        free_memory_of_all_array(arr_key);
        return NULL;
    }

    size_t len_tmp_text = strlen(encrypted_text);
    for (int i = 2; i < len_tmp_text; i += 3)
    {
        encrypted_text = paste_character_and_index(encrypted_text, i, ' ');
        len_tmp_text++;
    }

    free_memory_of_all_array(arr_key);
    return encrypted_text;
}

char* playfair_decrypt(const char* key, const char* text)
{
    if (!key || !text || strlen(text) == 0) return NULL;

    // Check if the key contains only alphabetic characters
    for (int i = 0; key[i] != '\0'; i++) {
        if (!isalpha(key[i]))
            return NULL;
    }

    char **arr_key = Create_key_word_array(key);
    if (!arr_key)
        return NULL;

    size_t text_len = strlen(text);
    char *tmp_text = (char*)calloc((text_len + 1), sizeof(char));
    if (!tmp_text)
    {
        free_memory_of_all_array(arr_key);
        return NULL;
    }

    strcpy(tmp_text, text);

    for (int i = 0; i < text_len; i++)
    {
        if (tmp_text[i] == ' ')
        {
            tmp_text = remove_char_index(tmp_text, i--);
            text_len--;
        }
    }

    for (int i = 0; i < text_len; i++)
    {
        if (tmp_text[i] == 'W')
        {
            free_memory_of_all_array(arr_key);
            free(tmp_text);
            return NULL;
        }
    }

    tmp_text = rewrite_cipher_with_matrix_key(arr_key, tmp_text, -1);
    free_memory_of_all_array(arr_key);
    return tmp_text;
}
