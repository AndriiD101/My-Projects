#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "playfair.h"
#include <ctype.h>

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