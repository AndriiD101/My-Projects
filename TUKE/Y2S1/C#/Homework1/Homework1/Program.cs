using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Homework1
{
    internal class Program
    {
        static bool IsPrime(int number)
        {
            if (number < 1) return false;
            for (int divisor = 2; divisor * divisor <= number; divisor++) 
            {
                if (number % divisor == 0)
                {
                    return false;
                }
            }
            return true;
        }

        //0 1 1 2 3 5 7 12
        static int Fibonacci(int number)
        { 
            int a = 0;
            int b = 1;
            if (number < 1) return 0;
            else if (number == 1) return a;
            else
            {
                for (int i = 2; i <= number; i++)
                {
                    int temp = a;
                    a = b;
                    b = temp + b;
                }
            }
            return b;
        }

        static bool IsTriangle(int a, int b, int c)
        {
            if(a+b >c || a+c>b ||  a+c<b) return true;
            return false;
        }

        static bool IsRightTriangle(int a, int b, int c)
        {
            if (a*a + b*b == c*c || a*a + c*c == b*b || a*a + c*c == b*b) return true;
            return false;
        }

        static void WriteMultiples(int number, int limit)
        {

            for (int i = 1; i <= limit; i++)
            {
                if (number % i <= 0) Console.WriteLine(i);
            }
        }

        static void WriteDivisors(int number)
        {

            for (int i = 1; i <= number; i++)
            {
                if (number % i <= 0) Console.WriteLine(i);
            }
        }

        static int DigitSum(int number)
        {
            int sum = 0;
            //int tmp_num = number;
            while (number > 0)
            {
                int digit = number % 10;
                sum += digit;
                number = (number - digit) / 10;
            }
            return sum;
            
        }

        static int DigitDiff(int number)
        {
            List<int> numbers = new List<int>();
            int max = 0;
            int min = int.MaxValue;
            while (number > 0)
            {
                int digit = number % 10;
                numbers.Add(digit);
                number = (number - digit) / 10;
            }
            foreach (int digit in numbers)
            {
                if (digit > max) max = digit;
                if (digit < min) min = digit;
            }
            return max-min;
        }

        static int Factorial(int number)
        {
            if (number == 1) return 1;
            else
            {
                return number * Factorial(number-1);
            }
        }

        static void WriteFibonacci(int n)
        {
            int a = 0;
            int b = 1;

            for (int i = 0; i < n; i++)
            {
                Console.Write(a + " ");
                int temp = a;
                a = b;
                b = temp + b;
            }
        }

        static void Main(string[] args)
        {
            Console.WriteLine(WriteMultiples(1g, 12);
            Console.ReadLine();
        }
    }
}
