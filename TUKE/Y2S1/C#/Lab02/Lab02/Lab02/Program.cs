using System.ComponentModel.Design;
using System.Runtime.CompilerServices;
using System.Security.Cryptography.X509Certificates;

namespace Lab02
{
    internal class Program
    {
        static char[ , ] LoadSudoku(string problem)
        {
            char[ , ] result = new char[9, 9];
            for (int row = 0; row < 9; row++) 
            {
                for(int col = 0; col < 9; col++)
                {
                    result[row, col] = problem[row * 9 + col];
                }
            }
            return result;
        }

        static void PrintSudoku(char[,] grid)
        {
            for (int row = 0; row < 9; row++)
            {
                if (row % 3 == 0)
                {
                    Console.WriteLine(" -----------------------");
                }
                for (int col = 0; col < 9; col++)
                {
                    if (col % 3 == 0)
                    {
                        Console.Write("| ");
                    }
                    if(grid [row, col] == '0')
                    {
                        Console.Write("." + " ");
                    }
                    else
                        Console.Write(grid[row, col] + " ");
                }
                Console.WriteLine("|");
            }
            Console.WriteLine(" -----------------------");
        }


        static bool IsValid(char[ , ] grid, int row, int col, char value)
        { 
            for(int rows  = 0; rows < 9; rows++)
            {
                if (grid [rows, col] == value)
                {
                    return false;
                }
            }
            for(int cols = 0; cols < 9; cols++)
            {
                if(grid [row, cols] == value)
                {
                    return false;
                }
            }
            int startRow = row / 3 * 3;
            int startCol = col / 3 * 3;
            for (int r = startRow; r < startRow + 3; r++)
            {
                for (int c = startCol; c < startCol + 3; c++)
                {
                    if (grid[r, c] == value)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        static bool SolveSudoku(char[,] grid)
        {
            for (int row = 0; row < 9; row++)
            {
                for (int col = 0; col < 9; col++)
                {
                    if (grid[row, col] == '0') 
                    {
                        for (char num = '1'; num <= '9'; num++)
                        {
                            if (IsValid(grid, row, col, num))
                            {
                                grid[row, col] = num;

                                if (SolveSudoku(grid))
                                {
                                    return true;
                                }

                                grid[row, col] = '0';
                            }
                        }
                        return false; 
                    }
                }
            }
            return true; 
        }


        static void FillSure(char[ , ] grid)
        {

        }

        static void Main(string[] args)
        {
            string example1 = "632005400004001300000000567000273005021406080000510000060030900048050002100029800";

            char[,] grid = LoadSudoku(example1);
            PrintSudoku(grid);

            Console.WriteLine("\n\n\n");

            SolveSudoku(grid);
            PrintSudoku(grid);
        }
    }
}