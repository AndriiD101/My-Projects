#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

void encode_char(const char character, bool bits[8])
{
    int letter_ascii = character;
    for(int div = 7; div >= 0; div--)
    {
        if(letter_ascii%2 == 0){
            bits[div] = letter_ascii%2;
            letter_ascii/=2;
        }
        else {
            bits[div] = letter_ascii%2;
            letter_ascii/=2;
        }
    }
}

char decode_byte(const bool bits[8])
{
    int result_ascii=0;
    int mid;
    for(int i=8; i>0; i--)
    {
        mid = pow(2, 8-i) * bits[i-1];
        result_ascii += mid;
    }
    return result_ascii;
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

void decode_bytes(const int rows, bool bytes[rows][8], char string[rows])
{
    int result_ascii=0;
    int mid;
    char ch;
    for(int i = 0; i < rows; i++)
    {
        for(int j=8; j>0; j--)
        {
            mid = pow(2, 8-j) * bytes[i][j-1];
            result_ascii += mid;
        }
        ch = result_ascii;
        string[i] = ch;
        result_ascii = 0;
    }
}


void bytes_to_blocks(const int cols, const int offset, bool blocks[offset*8][cols], const int rows, bool bytes[rows][8]) {
    for (int first = 0; first < offset; first++){
        for (int second=0; second < cols; second++){
            for (int third=0; third < 8; third++) {
                if(first*cols+second < rows) blocks[first*8+third][second] =  bytes[first*cols+second][third];
                else blocks[first*8+third][second] = 0;
            }
        }
    }
}

void blocks_to_bytes(const int cols, const int offset, bool blocks[offset*8][cols], const int rows, bool bytes[rows][8]) {
    for (int first=0; first < offset; first++){
        for (int second=0; second < cols; second++){
            for (int third=0; third< 8; third++) {
                if (first*cols+second > rows) return;
                bytes[first*cols+second][third] = blocks[first*8+third][second];
            }
        }
    }
}

int main()
{
    bool bits1[8];
    encode_char('B', bits1);
    for(int i = 0; i < 8; i++){
    printf("%d", bits1[i]);
    }
    printf("\n");
    // prints: 01000001

    bool bits2[8] = {0,1,0,0,0,0,1,0};
    printf("%c\n", decode_byte(bits2));
    // prints: A
    char* text = "Hello, how are you?";
    const int len = strlen(text);
    bool bytes1[len+1][8];
    encode_string(text, bytes1);
    for(int j = 0; j <= len; j++){
    printf("%c: ", text[j]);
    for(int i = 0; i < 8; i++){
        printf("%d", bytes1[j][i]);
    }
    printf("\n");
    }
    // prints:
    // H: 01001000
    // e: 01100101
    // l: 01101100
    // l: 01101100
    // o: 01101111
    // ,: 00101100
    //  : 00100000
    // h: 01101000
    // o: 01101111
    // w: 01110111
    //  : 00100000
    // a: 01100001
    // r: 01110010
    // e: 01100101
    //  : 00100000
    // y: 01111001
    // o: 01101111
    // u: 01110101
    // ?: 00111111
    // : 00000000

    bool bytes2[7][8] = {
    {0,1,0,0,1,0,0,0},
    {0,1,1,0,0,1,0,1},
    {0,1,1,0,1,1,0,0},
    {0,1,1,0,1,1,0,0},
    {0,1,1,0,1,1,1,1},
    {0,0,1,0,0,0,0,1},
    {0,0,0,0,0,0,0,0}
    };
    char string[7];
    decode_bytes(7, bytes2, string);
    printf("%s\n", string);
    // prints: Hello!
    int length = 4+1, cols = 3, offset = 2;
    bool bytes4[4+1][8] = {
        {0,1,0,0,0,0,0,1},
        {0,1,1,0,1,0,0,0},
        {0,1,1,0,1,1,1,1},
        {0,1,1,0,1,0,1,0},
        {0,0,0,0,0,0,0,0}
    };
    bool blocks1[offset*8][cols];
    bytes_to_blocks(cols, offset, blocks1, length, bytes4);
    for(int j = 0; j < offset*8; j++){
        for(int i = 0; i < cols; i++){
        printf("%d ", (blocks1[j][i] == true) ? 1 : 0);
    }
    printf("\n");
    if(j % 8 == 7){
        printf("\n");
    }
    }
    bool blocks2[2*8][3] = {
    {0,0,0},
    {1,1,1},
    {0,1,1},
    {0,0,0},
    {0,1,1},
    {0,0,1},
    {0,0,1},
    {1,0,1},
    {0,0,0},
    {1,0,0},
    {1,0,0},
    {0,0,0},
    {1,0,0},
    {0,0,0},
    {1,0,0},
    {0,0,0}
    };
    bool bytes9[length][8];
    blocks_to_bytes(3, 2, blocks2, length, bytes9);
    for(int j = 0; j < length; j++){
        for(int i = 0; i < 8; i++){
            printf("%d", bytes9[j][i]);
        }
        printf("\n");
    }
// prints:
// 01000001
// 01101000
// 01101111
// 01101010
// 00000000
    return 0;
    
}