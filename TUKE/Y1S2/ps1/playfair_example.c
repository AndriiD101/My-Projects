//#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "playfair.h"
#include <ctype.h>

char* set_difference(char* A, char* B); // A \ B (A i B множини)
char** create_arr_key(const char* key); // створю масив з ключа
bool letter(char ch); //вертає тру якшо була параметр буква
char* insert(char* arr, int position, char ch); // вставляє в масив символів задану букву на задану позицію
char* insert_X(const char* text); //вставляє в потрібному місці(див. правила шиврування Плейфаєра) Х / змінює W на V / прибирає непотрібні символи / рвсі букви стають toupper();
char* rewrite_cipher_with_matrix_key(char** key, char* text, int direction); //вертає кінечний зашифрований рядок на основі ключа-масиву та рядка для шифрування ***якщо direction додатнє, ф-я шифрує, якщо відємний то дешифрує
void seek_letter(char **arr, char ch, int *x, int *y); //на позиції х та у записує координати шуканої літери в масиві
void swap(int *a, int *b); //
char* del(char* text, int index); //
char* clean_sign(const char* text); //вертає лише масив великих букв
char* remove_duplicate(const char* text);
void free_arr(char **arr);

int main(){
	char *encrypted, *decrypted;

	// even length of string
	encrypted = playfair_encrypt("SeCReT", "Hello world");
	printf("%s\n", encrypted);
	// "Hello world" --> "HELXLOVORLDX"
	// IS JZ JQ XN TK JC
	decrypted = playfair_decrypt("SeCReT", encrypted);
	printf("%s\n", decrypted);
	// HELXLOVORLDX
	free(encrypted);
	free(decrypted);

	// odd length of string
	encrypted = playfair_encrypt("world", "Hello");
	printf("%s\n", encrypted);
	// "Hello" --> "HELXLO"
	// JB RY DR
	decrypted = playfair_decrypt("world", encrypted);
	printf("%s\n", decrypted);
	// HELXLO
	free(encrypted);
	free(decrypted);

	// letter 'X' in message
	encrypted = playfair_encrypt("Password", "Taxi please");
	printf("%s\n", encrypted);
	// "Taxi please" --> "TAXIPLEASE"
	// UP YH AK DO OB
	decrypted = playfair_decrypt("Password", encrypted);
	printf("%s\n", decrypted);
	// TAXIPLEASE
	free(encrypted);
	free(decrypted);

	// multi 'X's in message
	encrypted = playfair_encrypt("please", "Taxxxiii");
	printf("%s\n", encrypted);
	// "Taxxxiii" --> "TAXXXIIXIX"
	// RS EE VJ JV JV
	decrypted = playfair_decrypt("please", encrypted);
	printf("%s\n", decrypted);
	// TAXXXIIXIX
	free(encrypted);
	free(decrypted);
}

char* playfair_encrypt(const char* key, const char* text)
{
	if (!key && !text)
	{
        return NULL;
    }
    char **arr_key = create_arr_key(key);// free() don't
	char *tmp_text;
	tmp_text = insert_X(text);// free() don't. free() in down
	tmp_text = rewrite_cipher_with_matrix_key(arr_key, tmp_text, 1);// free() don't. free() in down  tmp_text free() do
	size_t len_tmp_text = strlen(tmp_text);
		for (int i = 2; i < len_tmp_text; i += 3)
		{
			tmp_text = insert(tmp_text, i, ' ');// free() do
			len_tmp_text++;
		}
	free_arr(arr_key);
	return tmp_text;

}


char* playfair_decrypt(const char* key, const char* text)
{
	if (!key && !text)
	{
		return NULL;
	}
	char **arr_key = create_arr_key(key);// free() don't. free() in down
		size_t len_text = strlen(text);
		char *tmp_text = (char*)calloc((len_text + 1), sizeof(char));
		strcpy(tmp_text, text);
		for (int i = 0; i < len_text; i++)
		{
			if (tmp_text[i] == ' ')
			{
				tmp_text = del(tmp_text, i--);//free() do
				len_text--;
			}
			if (tmp_text[i] == 'W')
			{
				free_arr(arr_key);
				free(tmp_text);
				return NULL;
			}
		}
		tmp_text = rewrite_cipher_with_matrix_key(arr_key, tmp_text, -1);// free() don't. free() in down
		free_arr(arr_key);
		return tmp_text;
}

// free() don't. free() in down  tmp_text free() do
char* rewrite_cipher_with_matrix_key(char** key, char* text, int direction)
{
	int tmp_direction = 0;
	if (direction >= 0)
	{
		tmp_direction = 1;
	}
	else
	{
		tmp_direction = -1;
	}
	int row1, column1,
		row2, column2;
	row1 = column1 = row2 = column2 = 0;
	size_t len_text = strlen(text);
	if (!(len_text % 2))
	{
		char *encrypt_text = (char*)calloc((len_text + 1), sizeof(char));
		for (int i = 1; i < len_text; i += 2)
		{
			seek_letter(key, text[i - 1], &column1, &row1);//free() don't
			seek_letter(key, text[i], &column2, &row2);//free() don't
			if (column1 == column2)
			{
				row1 = ((row1 + tmp_direction) % 5);
				row2 = ((row2 + tmp_direction) % 5);
			}
			else if (row1 == row2)
			{
				column1 = ((column1 + tmp_direction) % 5);
				column2 = ((column2 + tmp_direction) % 5);
			}
			else
			{
				swap(&column1, &column2);
			}
			encrypt_text[i - 1] = key[row1 >= 0 ? row1 : 4][column1 >= 0 ? column1 : 4];
			encrypt_text[i] = key[row2 >= 0 ? row2 : 4][column2 >= 0 ? column2 : 4];
		}
		encrypt_text[len_text] = '\0';
		free(text);
		return encrypt_text;
	}
	free(text);
}

void swap(int *a, int *b)
{
	int tmp = *a;
	*a = *b;
	*b = tmp;
}

//free() don't
void seek_letter(char **arr, char ch, int *x, int *y)
{
	*x = -1;
	*y = -1;
	char seek_char = ch;
	seek_char = toupper(seek_char);
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			if (arr[i][j] == seek_char)
			{
				*x = j;
				*y = i;
				return;
			}
		}
	}

}

/**///free() do
char* del(char* text, int index)
{
	size_t len_text = strlen(text) + 1;
	if (index >= len_text - 1)
	{
		return text;
	}
	size_t len_tmp_text = len_text - 1;
	char *tmp_text = (char*)calloc(len_tmp_text, sizeof(char));
	for (int i = 0; i < index; i++)
	{
		tmp_text[i] = text[i];
	}
	for (int i = index; i < len_tmp_text - 1; i++)
	{
		tmp_text[i] = text[i + 1];
	}
	tmp_text[len_tmp_text - 1] = '\0';
	free(text);
	return tmp_text;
}

/**/ // free() don't
char* insert_X(const char* text)
{
	char *tmp_text = clean_sign(text);//free() don't
	for (int i = 1;; i += 2)
	{
		if (!tmp_text[i])
		{
			tmp_text = insert(tmp_text, i, 'X');
			break;
		}
		else if (tmp_text[i])
		{
			if (tmp_text[i - 1] == tmp_text[i] && tmp_text[i] != 'X')
			{
				tmp_text = insert(tmp_text, i, 'X');//free() do
			}
		}
		if (!tmp_text[i + 1])
		{
			break;
		}
	}
	return tmp_text;
}

/**/ // free() don't
char* clean_sign(const char* text)
{
    if (!text)
    {
        return NULL;
    }

    size_t len_text = strlen(text);
    char* tmp_text = (char*)calloc(len_text + 1, sizeof(char));
    if (!tmp_text) return NULL; // Check if allocation fails

    size_t len_cleaned_text = 0; // Tracks the length of cleaned text
    for (size_t i = 0; i < len_text; i++)
    {
        char current_char = toupper(text[i]);

        // Replace 'W' with 'V'
        if (current_char == 'W')
        {
            current_char = 'V';
        }

        // Check if the character is a letter
        if (isalpha(current_char))
        {
            tmp_text[len_cleaned_text++] = current_char;
        }
    }

    tmp_text[len_cleaned_text] = '\0'; // Add null terminator
    return tmp_text;
}


/**///free() do
char* clean_key(char *key)
{
	if (!key)
	{
		return NULL;
	}
	size_t len_key = strlen(key);
	char * new_key = (char*)calloc((len_key + 1), sizeof(char));
	for (int i = 0; i < len_key; i++)
	{
		new_key[i] = toupper(key[i]);
		if (!letter(new_key[i]) && new_key[i] != ' ')
		{
			free(key);
			free(new_key);
			return NULL;
		}
		if (new_key[i] == 'W')
		{
			new_key[i] = 'V';
		}
	}
	for (int i = 0; i < len_key; i++)
	{
		if (new_key[i] == ' ')
		{
			new_key = del(new_key, i--);//free(); do
			len_key--;
		}
	}
	new_key[len_key] = '\0';
	free(key);
	return new_key;
}

bool letter(char ch)
{
	return toupper(ch) >= 'A' && toupper(ch) <= 'Z' ? true : false;
}

//free() do
char* insert(char* arr, int position, char ch)
{
	size_t len_arr = strlen(arr);
	if (position <= len_arr)
	{
		char * new_arr = (char*)calloc(len_arr + 2, sizeof(char));
		for (int i = 0; i < position && i < len_arr; i++)
		{
			new_arr[i] = arr[i];
		}
		new_arr[position] = ch;
		for (int i = position; i < len_arr; i++)
		{
			new_arr[i + 1] = arr[i];
		}
		new_arr[len_arr + 1] = '\0';
		free(arr);
		return new_arr;
	}
	return arr;
}

char* set_difference(char* A, char* B) // A \ B
{
	size_t len_A;
	size_t len_B;
	if (A)
	{
		len_A = strlen(A);
	}
	else
	{
		len_A = 0;
	}
	if (B)
	{
		len_B = strlen(B);
	}
	else
	{
		len_B = 0;
	}
	char *C = (char*)calloc(len_A + 1, sizeof(char));
	int t = 0;
	for (int i = 0; i < len_A; i++, t++)
	{
		C[t] = A[i];
		for (int j = 0; j < len_B; j++)
		{
			if (toupper(C[t]) == toupper(B[j]))
			{
				t--;
				break;
			}
		}
	}
	C[t] = '\0';
	free(A);
	return C;
}

/**/// free() don't. free() in down
char** create_arr_key(const char* key)
{
	size_t len_key = strlen(key);
	size_t len_alpha = strlen(ALPHA);
	char *alpha = (char*)calloc(len_alpha + 1, sizeof(char));
	strcpy(alpha, ALPHA);
	char *short_key = (char*)calloc(len_key + 1, sizeof(char));
	char *free_ptr = NULL;
	free_ptr = remove_duplicate(key);
	strcpy(short_key, free_ptr);
	free(free_ptr); free_ptr = NULL;
	short_key = clean_key(short_key);//free() do
	if (short_key && *short_key)
	{
		size_t len_short_key = strlen(short_key);
		alpha = set_difference(alpha, short_key);//free(alpha) do;
		char **arr_key = (char**)calloc(5, sizeof(char*));
		for (int i = 0; i < 5; i++)
		{
			arr_key[i] = (char*)calloc(5, sizeof(char));
		}
		for (int i = 0, k = 0; (i < 5); i++)
		{
			for (int j = 0; (j < 5); j++)
			{
				if (k < len_short_key)
				{
					if (letter(short_key[k]))
					{
						arr_key[i][j] = toupper(short_key[k++]);
					}
					else
					{
						free(alpha);
						free(short_key);
						free_arr(arr_key);
						return NULL;
					}
				}
				else
				{
					arr_key[i][j] = toupper(alpha[k++ - len_short_key]);
				}
			}
		}
		free(alpha);
		free(short_key);
		return arr_key;
	}
	
	free(alpha);
	if (short_key)
	{
		free(short_key);
	}
}


/**/// free() don't
char* remove_duplicate(const char* text)
{
    size_t len_text = strlen(text);
    char* tmp_text = (char*)calloc(len_text + 1, sizeof(char));
    if (!tmp_text) return NULL; // Check if allocation fails

    size_t len_tmp_text = 0; // Tracks the length of modified text
    for (size_t i = 0; i < len_text; i++)
    {
        char current_char = toupper(text[i]);

        // Replace 'W' with 'V'
        if (current_char == 'W')
            current_char = 'V';

        // Check if the character is a duplicate
        int is_duplicate = 0;
        for (size_t j = 0; j < len_tmp_text; j++)
        {
            if (toupper(tmp_text[j]) == current_char)
            {
                is_duplicate = 1;
                break;
            }
        }

        // If not a duplicate, add it to the modified text
        if (!is_duplicate)
            tmp_text[len_tmp_text++] = current_char;
    }
    tmp_text[len_tmp_text] = '\0'; // Add null terminator
    return tmp_text;
}

/**/
void free_arr(char **arr)
{
	for (int i = 0; i < 5; i++)
	{
		free(arr[i]);
	}
	free(arr);
}