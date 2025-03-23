using System;
using System.Collections.Generic;

namespace Command
{
    internal class Program
    {
        //(code) https://www.dofactory.com/net/command-design-pattern
        // The 'Command' abstract class
        public abstract class Command
        {
            public abstract void Execute();
            public abstract void UnExecute();
        }

        // The 'ConcreteCommand' class
        public class CalculatorCommand : Command
        {
            char @operator;
            int operand;
            Calculator calculator;

            public CalculatorCommand(Calculator calculator, char @operator, int operand)
            {
                this.calculator = calculator;
                this.@operator = @operator;
                this.operand = operand;
                Console.WriteLine($"Created command: {@operator} {operand}");
            }

            public char Operator
            {
                set { @operator = value; }
            }

            public int Operand
            {
                set { operand = value; }
            }

            public override void Execute()
            {
                Console.WriteLine($"Executing command: {@operator} {operand}");
                calculator.Operation(@operator, operand);
            }

            public override void UnExecute()
            {
                Console.WriteLine($"Unexecuting command: Undo {@operator} {operand}");
                calculator.Operation(Undo(@operator), operand);
            }

            private char Undo(char @operator)
            {
                switch (@operator)
                {
                    case '+': return '-';
                    case '-': return '+';
                    case '*': return '/';
                    case '/': return '*';
                    default:
                        throw new ArgumentException("@operator");
                }
            }
        }

        // The 'Receiver' class
        public class Calculator
        {
            int curr = 0;

            public void Operation(char @operator, int operand)
            {
                Console.WriteLine($"Calculator performing operation: {@operator} {operand}");
                switch (@operator)
                {
                    case '+': curr += operand; break;
                    case '-': curr -= operand; break;
                    case '*': curr *= operand; break;
                    case '/': curr /= operand; break;
                }
                Console.WriteLine($"Current value = {curr,3} (after {@operator} {operand})");
            }
        }

        // The 'Invoker' class
        public class User
        {
            Calculator calculator = new Calculator();
            List<Command> commands = new List<Command>();
            int current = 0;

            public void Redo(int levels)
            {
                Console.WriteLine($"\n---- Redo {levels} levels ----");
                for (int i = 0; i < levels; i++)
                {
                    if (current < commands.Count)
                    {
                        Console.WriteLine($"Redoing command at index {current}");
                        Command command = commands[current++];
                        command.Execute();
                    }
                    else
                    {
                        Console.WriteLine("No more commands to redo.");
                    }
                }
            }

            public void Undo(int levels)
            {
                Console.WriteLine($"\n---- Undo {levels} levels ----");
                for (int i = 0; i < levels; i++)
                {
                    if (current > 0)
                    {
                        Console.WriteLine($"Undoing command at index {current - 1}");
                        Command command = commands[--current];
                        command.UnExecute();
                    }
                    else
                    {
                        Console.WriteLine("No more commands to undo.");
                    }
                }
            }

            public void Compute(char @operator, int operand)
            {
                Console.WriteLine($"\nUser computing: {@operator} {operand}");
                Command command = new CalculatorCommand(calculator, @operator, operand);
                command.Execute();
                commands.Add(command);
                current++;
            }
        }

        // Command Design Pattern
        public static void Main(string[] args)
        {
            User user = new User();

            // User presses calculator buttons
            user.Compute('+', 100);
            user.Compute('-', 50);
            user.Compute('*', 10);
            user.Compute('/', 2);

            // Undo 4 commands
            user.Undo(4);

            // Redo 3 commands
            user.Redo(3);

            Console.ReadKey();
        }
    }
}
