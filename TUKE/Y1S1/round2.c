#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>

int is_whites(const char c)
{
    if(c == ' ' || c == '\t' || c == '\n')
    {
        return 1;
    }
    return 0;
}

void change_whites(char string[])
{
    for(int i = 0; i < strlen(string); i++)
    {
        if(string[i] == ' ' || string[i] == '\n')
        {
            string[i] = '.';
        }
    }
}

int guess_evaul(const int guess, const int my_number)
{
    if(guess == my_number)
    {
        return 1;
    }
    else if(guess < my_number)
    {
        return 2;
    }
    else
    {
        return 0;
    }
}

int leap_year(const int year)
{
    if(year<1 && year<4443)
    {
        return -1;
    }
    if(year % 400 == 0)
    {
        return 1;
    }
    else if(year % 100 == 0)
    {
        return 0;
    }
    else if(year % 4 == 0)
    {
        return 1;
    }
    return 0;
}

int count_positives(const int size, const int array[])
{
    int flag = 0;
    if(array == NULL)
    {
        return -1;
    }
    for(int i=0; i<size; i++)
    {
        if(array[i]>0)
        {
            flag ++;
        }
    }
    return flag;
}

int count_whites(const char string[])
{
    int count = 0;
    for(int i = 0; i < strlen(string); i++)
    {
        if (string[i] == ' ' || string[i] == '\n' || string[i] == '\t') 
        {
            count++;
        }
    }
    return count;
}

int direction_correction(const int degree)
{
    int degree_l = degree;
    if(degree < 0 || degree%90!=0)
    {
        return -1;
    }
    // if(degree == 0 || degree == 360)
    // {
    //     return 360;
    // }
    // else if (degree == 90)
    // {
    //     return 90;
    // }
    // else if (degree == 180)
    // {
    //     return 180;
    // }
    // else if (degree == 270)
    // {
    //     return 270;
    // }
    while(degree_l>=360)
    {
        degree_l -= 360;
    }
    return degree_l;
}

int all_positives(const int size, const int array[])
{
    for(int i = 0; i < size; i++)
    {
        if(array[i] <= 0)
        {
            return 0;
        }
    }
    return 1;
}

int last_positive(const int size, const int array[])
{
    int tmp = 0;
    if(array == NULL)
    {
        return -1;
    }
    for(int i = 0; i < size; i++)
    {
        if(array[i] > 0)
        {
            tmp = array[i];
        }
    }
    if(tmp > 0)
    {
        return tmp;
    }
    return -1;
}

int bynary_num(const int num)
{
    if(num == 0 || num == 1)
    {
        return 1;
    }
    if(-1000<num && 1000>num )
    {
        return 0;
    }
    return -1;
}

void swap_sign(const int size, int array[])
{
    for(int i=0; i<size; i++)
    {
        array[i] = array[i] * -1;
    }
}

int div_by_3(const int num)
{
    if(num%3 == 0)
    {
        return 1;
    }
    return 0;
}

int same_case(const char a, const char b) {
    if (!isalpha(a) || !isalpha(b)) {
        return -1;
    }

    if ((isupper(a) && isupper(b)) || (islower(a) && islower(b))) {
        return 1;
    }

    return 0;
}

int find_first_A(const char string[])
{
    for(int i = 0; i <strlen(string); i++) 
    {
        if(string[i] == 'A' || string[i] == 'a')
        {
            return i;
        }
    }
    return 0;
}

int main()
{
    printf("%d %d \n", is_whites('0'), is_whites(' '));
    ////////////////////////////////////////////////////////////////
    char string[100] = "Hello world";
    change_whites(string);
    printf("%s\n", string);
    ////////////////////////////////////////////////////////////////
    printf("%d %d %d\n", guess_evaul(34, 22), guess_evaul(22, 34), guess_evaul(34,34));
    ////////////////////////////////////////////////////////////////
    printf("%d %d %d\n", leap_year(4000), leap_year(3000), leap_year(3004));
    ////////////////////////////////////////////////////////////////
    const int array1[] = {1,2,0,3,4,0};
    const int array2[] = {1,2,6,3,4,7};
    const int array3[] = {-1,-2,0,-3,0,-2};
    printf("%d %d %d\n", count_positives (6, array1), count_positives (6, array2), count_positives (6, array3));
    ////////////////////////////////////////////////////////////////
    const char string2[] = "Hello, how are you?\n";
    printf("%d\n", count_whites(string2));
    ////////////////////////////////////////////////////////////////
    printf("lol heere>>> ");
    printf("%d %d %d\n", direction_correction (-90), direction_correction (540), direction_correction (90));
    ////////////////////////////////////////////////////////////////
    const int array4[] = {1,2,0,3,4,0}; 
    const int array5[] = {1,2,6,3,4,7}; 
    const int array6[] = {1,2,-1,3,4,-2};
    printf("%d %d %d\n", all_positives (6, array1), all_positives (6, array2), all_positives (6, array3)); 
    // 0 1 0
    ////////////////////////////////////////////////////////////////
    const int array7[] = {0,1,0};
    const int array8[] = {-1,0,6,-2};
    printf("%d %d\n", last_positive (3, array7), last_positive(4, array8)); 
    // 1 -1
    ////////////////////////////////////////////////////////////////
    printf("%d %d %d\n", bynary_num (-1001), bynary_num (3), bynary_num (1));
    ////////////////////////////////////////////////////////////////
    int array[] = {1,2,0,-3,4,0}; swap_sign (6, array);
    for (int i = 0; i < 6; i++){ 
        printf("%d ", array[i]);
    }
    printf("\n");
    // -1 -2 0 3 -4 0
    ////////////////////////////////////////////////////////////////
    printf ("%d %d %d\n", div_by_3(-3), div_by_3(6), div_by_3(8));
    ////////////////////////////////////////////////////////////////
    printf("%d %d %d\n", same_case('a', 'f'), same_case ('L','g'), same_case ('#', 'P')); // 1 0-1
    ////////////////////////////////////////////////////////////////
    printf("%d\n", find_first_A("Tommorow afternoon"));
    
    return 0;
}