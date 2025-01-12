#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int sum_of_digits(unsigned long long x){
    int result = 0;
    if(x!=0){
        result = x%10+sum_of_digits(x/10);
        // printf("%d\n", result);
    }
    return result;
}

int main(){
    unsigned long long input = 0;
    scanf("%lld",&input);
    while(input>=10){
        input = sum_of_digits(input);
    }
    printf("%lld\n", input);
    return 0;
}   