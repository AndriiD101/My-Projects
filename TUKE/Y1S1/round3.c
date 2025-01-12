#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

int max_2d(const int size, const int array[][size])
{
    int max = 0;
    if(array == NULL) return -1;
    for(int row = 0; row < size; row++){
        for(int col = 0; col < size; col++){
            if(array[row][col] > max)
            {
                max = array[row][col];
            }
        }
    }
    return max;
}

int vowel_count_2d(const int row, const int col, char string[][col])
{
    if(string == NULL) return -1;
    int count = 0;
    for(int l_row = 0; l_row < row; l_row++)
    {
        for(int l_col = 0; l_col < col; l_col++)
        {
            if(tolower(string[l_row][l_col]) == 'a' || tolower(string[l_row][l_col]) == 'e' || tolower(string[l_row][l_col]) == 'i' || tolower(string[l_row][l_col]) == 'o' || tolower(string[l_row][l_col]) == 'u'|| tolower(string[l_row][l_col]) == 'y')
            {
                count++;
            }
        }
    }
    return count;
}

int is_in_array_2d(const int num, const int size, int array[][size])
{
    if(array == NULL) return -1;
    
    for(int row=0; row < size; row++) {
        for(int col=0; col < size; col++){
            if(array[row][col] == num) return 1;
        }
    }
    return 0;
}

int largest_line(const int size, int array[][size])
{
    int max_sum = 0;
    int max_row = 0;
    if(array == NULL) return -1;
    for(int row = 0; row < size; row++){
        int sum = 0;
        for(int col = 0; col < size; col++){
            sum += array[row][col];
            if(sum > max_sum){
                max_row = row;
                max_sum = sum;
            }
        }
    }
    return max_row;
}

void swap_case_2d(const int row, const int col, char string[][col]){
    if(string == NULL) return;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (isupper(string[i][j])) {
                string[i][j] = tolower(string[i][j]);
            } else {
                string[i][j] = toupper(string[i][j]);
            }
        }
    }
}

int largest_col(const int size, int array[][size]){
    int max_sum = 0;
    int max_col = 0;
    if(array == NULL) return -1;
    
    for(int col = 0; col < size; col++){
        int sum = 0;
        for(int row = 0; row < size; row++){
            sum += array[row][col];
            if(sum > max_sum){
                max_col = col;
                max_sum = sum;
            }
        }
    }
    return max_col;
}

int count_zero(const int size, int array[][size]){
    int count_zero = 0;
    if(array == NULL) return -1;

    for(int row = 0; row < size; row++){
        for(int col = 0; col < size; col++){
            if(array[row][col] == 0) count_zero++;
        }
    }
    return count_zero;
}

void swap_sign_2d(const int size, int array[][size]){
    if(array == NULL) return;
    for(int row = 0; row < size; row++){
        for(int col = 0; col < size; col++){
            array[row][col] *= -1;
        }
    }
}

int main()
{
    int array[2][2]={{1, 0},
                     {0, -3}};
    printf("%d\n", max_2d(2, array));
    printf("----------------------------------------------------------------\n");
    char string[3][50] ={"hello WORLD", "aHOJ", "Ahoj"};
    printf("%d\n", vowel_count_2d(3, 50, string));
    printf("----------------------------------------------------------------\n");
    int array1[2][2] = { {1,0}, {0,-3} };
    printf("%d %d\n", is_in_array_2d (2, 2, array1), is_in_array_2d(-3, 2, array1)); 
    // 0 1
    printf("----------------------------------------------------------------\n");
    int array_line[2][2] = { {1,0}, {0,-3} }; 
    printf("%d\n", largest_line (2, array_line)); // 0
    printf("----------------------------------------------------------------\n");
    swap_case_2d(3, 50, string);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 50; j++) {
            printf("%c", string[i][j]);
        }
        printf("\n");
    }
    printf("----------------------------------------------------------------\n");
    int array_col[2][2] = { {1,20}, 
                            {0,-3} }; 
    printf("%d\n", largest_col(2, array_col)); //0
    printf("----------------------------------------------------------------\n");
    printf("%d\n", count_zero(2, array)); //0
    printf("----------------------------------------------------------------\n");
    int array_swap[2][2] = { {1,20}, 
                            {-5,-3} };
    swap_sign_2d(2, array_swap);
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 2; j++){
            printf("%d ", array_swap[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}
