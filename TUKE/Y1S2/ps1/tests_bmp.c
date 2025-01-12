#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>
#include "bmp.h"

extern char* caesar_encrypt(const char* text, const int step);

void test_when_valid_string_is_given_then_return_encrypted_string();
void test_when_null_is_given_then_return_null();
void test_when_empty_string_is_given_then_return_empty_string();
void test_when_negative_step_return_null();
void test_when_step_is_zero_return_some_string();
void test_when_correct_step_is_given_then_return_some_string();

const char words[10][10] = { "SEPTEMBER", "THOUSAND", "CASSOVIA", "NEPAL", "EVEREST",
                            "CAESAR", "ZONE", "YES", "ANDERSON", "KEY" };
const char en_words[10][10] = { "XJUYJRGJW", "YMTZXFSI", "HFXXTANF", "SJUFQ", "JAJWJXY",
                                "HFJXFW", "ETSJ", "DJX", "FSIJWXTS", "PJD" };
int main(){
    printf("Tests started...\n");
    test_when_valid_string_is_given_then_return_encrypted_string();
    test_when_null_is_given_then_return_null();
    test_when_empty_string_is_given_then_return_empty_string();
    test_when_negative_step_return_null();
    test_when_step_is_zero_return_some_string();
    test_when_correct_step_is_given_then_return_some_string();
    printf("All tests passed\n");
}

void test_when_valid_string_is_given_then_return_encrypted_string(){
    printf("Test when valid string is given then return encrypted string\n");
   
    for(int idx=0;idx<10;idx++){
        char* result = caesar_encrypt(words[idx],5);
        assert(strcmp(result,en_words[idx]) ==0);
        free(result);
    }
    printf("Passed\n");
}

void test_when_null_is_given_then_return_null(){
    printf("Test when null must return null\n");
    char* result = caesar_encrypt(NULL,5);
    assert(result == NULL);
    printf("Passed\n");
}

void test_when_empty_string_is_given_then_return_empty_string(){
    printf("Test when empty string\n");
    char* result = caesar_encrypt("",5);
    assert(strlen(result) == 0);
    printf("Passed\n");
}

void test_when_negative_step_return_null(){
    printf("Test when negative step is given return\n");
    char* result = caesar_encrypt("HELLO",-1);
    assert(result==NULL);
    printf("Passed\n");
}

void test_when_step_is_zero_return_some_string(){
    printf("Test when step is zero return some str\n");
    char* result = caesar_encrypt("HELLO",0);
    assert(result != NULL);
    printf("Passed\n");
}

void test_when_correct_step_is_given_then_return_some_string(){
    printf("Test when correct step and return some str\n");
    char* result = caesar_encrypt("HELLO",0);
    assert(result != NULL);
    printf("Passed\n");
}







char* caesar_encrypt(const char* text, int step){
    int len = strlen(text);
    char* result = (char*)calloc(len + 1, sizeof(char));

    for( int index = 0; index < len; index++ ){
        result[index] = (text[index] - 'A' + step) % 26 + 'A';
    }
    result[len] = '\0';

    return result;
}

// void test_encrypt_decrypt(){

// }
