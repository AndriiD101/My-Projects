#include <stdio.h>

//Task1
float lift_a_car(const int stick_length, const int human_weight, const int car_weight)
{
    float m = (float) stick_length * (float) human_weight;
    float k = (float) human_weight+(float)car_weight;
    float result = m/k;
    float rounded = ((int)(result *100+0.5))/100.0;
        return rounded;
}

//Task2
float unit_price(const float pack_price, const int rolls_count, const int pieces_count)
{
    float m = pack_price *100;
    float k = rolls_count * pieces_count;
    float result = m/k;
    float rounded =((int)(result*100+0.5))/100.0;
        return rounded;
}
//Task3
int bank_notes(const int price)
{
    int i=0;
    int pr = price;
    if(price<10)
    {
        return -1;
    }
    else
    {
        while(pr>=200)
        {
            i++;
            pr-=200;
        }
        while(pr>=100)
        {
            i++;
            pr-=100;
        }
        while(pr>=50)
        {
            i++;
            pr-=50;
        }
        while(pr>=20)
        {
            i++;
            pr-=20;
        }
        while(pr>=10)
        {
            i++;
            pr-=10;
        }
    }
    return i;
}

//Task4
int euler(int const n) {
    int result = n;
    int c = n;
    for (int p = 2; p * p <= n; ++p) {
        if (c%p==0) {
            while (c%p==0) {
                c/=p;
            }
            result=result-result/p;
        }
    }

    if (c > 1) {
        result=result-result / c;
    }

    return result;
}

//Task5

void swap(int *xp, int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}
 
void selectionSort(int arr[], int n)
{
    int i, j, min_idx;
    for (i = 0; i < n-1; i++)
    {
        min_idx = i;
        for (j = i+1; j < n; j++)
          if (arr[j] < arr[min_idx])
            min_idx = j;
        if(min_idx != i)
            swap(&arr[min_idx], &arr[i]);
    }
}

int find_missing_number(const int n, const int arr[])
{
    int arr1[n];
    for (int i =0;i<n; i++)
        arr1[i]=arr[i];
    selectionSort(arr1, n);
    for(int i=0; i<n+1; i++)
    {
        if(arr1[i]!=i)
            return i;
    }
    return 0;
}
//Task6
unsigned long sum_squared(const int line)
{
  unsigned long n = line;
  unsigned long f = n+1;
  unsigned long arr[n];
  unsigned long  result = 0;

  if(n==0)
  {
    printf("1");
  }
  else
  {
        int coef = 1; 
    arr[0] = coef;
        for (int i = 1; i <= f; i++)
    {
            //printf("%4d", coef); 
            coef = coef * (f - i) / i;
      arr[i] = coef;
        }
    for (int i = 0; i < f; i++)
    result += arr[i]*arr[i];
  }
        return result;
}
//Task7
int array_min(const int input_array[], const int array_size)
{
    int arr1[array_size];
    for(int i = 0; i<array_size; i++)
        arr1[i] = input_array[i];
    if(input_array == NULL)
        return -1;
    else
    {
       selectionSort(arr1, array_size);
    }
    return arr1[0];
}

int array_max(const int input_array[], const int array_size)
{
    int arr1[array_size];
    if(input_array == NULL)
        return -1;
    else
    {
        for(int i = 0; i<array_size; i++)
            arr1[i] = input_array[i];
        selectionSort(arr1, array_size);
    }
    return arr1[array_size-1];
}

//Task8
//maybe i should make this better, because i am not sure if
//this will work for all numbers
int factorize_count(const int n)
{
    int tmp = n;
    //int arr[5];
    int count = 0;
    for(int i=2; i<tmp; i++)
    {
        if(tmp%i==0)
        {
            count++;
            tmp/=i;
        }
    }
    return count+1;
}

//Task9
void podium(const int n, int arr[])
{
    float second = n/3;
    if(n%3!=0)
        second+=1;
    arr[0]=second;
    float first = second+1;
    float third = n-first-second;
    arr[1] = first;
    arr[2] = third;

}

int main()
{
    printf("----------------------------------------------\n");
    //Task1
    float first_result = lift_a_car(2, 91, 3243);
    printf("%.4f\n", first_result);
    printf("----------------------------------------------\n");
    //Task2
    float second_result = unit_price(5.46, 22, 110);
    printf("%.4f\n", second_result);
    printf("----------------------------------------------\n");
    //Task3
    int third_result = bank_notes(540);
    printf("%d\n", third_result);
    printf("----------------------------------------------\n");
    //Task4
    printf("%d\n", euler(190));
    // prints: 2
    printf("%d\n", euler(191));
    // prints: 4
    printf("----------------------------------------------\n");
    //Task5
    int input_array[] = {2,1,0};
    int fiveth_result = find_missing_number(3, input_array);
    printf("%d\n", fiveth_result);
    printf("----------------------------------------------\n");
    //Task6
    printf("%lu\n", sum_squared(1));
    // prints: 2
    printf("%lu\n", sum_squared(4));
    // prints: 70
    printf("%lu\n", sum_squared(33));
    // prints: 7219428434016265740
    printf("----------------------------------------------\n");
    //Task7
    int InputArray[] = {1,2,3,4,5};
    printf("%d\n", array_min(InputArray, 5));
    printf("%d\n", array_max(InputArray, 5));
    printf("%d\n", array_max(NULL, 5));
    printf("----------------------------------------------\n");
    //Task8
    printf("%d\n", factorize_count(12));
    printf("----------------------------------------------\n");
    //Task9
    int heights[3];
    int material = 180;
    podium(material, heights);

    for(int i = 0; i < 3; i++){
        printf("%d ", heights[i]);
    }
    printf("\n");
    // prints: 2 3 1
    printf("----------------------------------------------\n");
    return 0;
}
