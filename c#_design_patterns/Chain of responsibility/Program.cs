using System;

namespace Chain.RealWorld
{
    //https://www.youtube.com/watch?v=FafNcoBvVQo
    //code : https://www.dofactory.com/net/chain-of-responsibility-design-pattern
    // Class holding request details
    public class Purchase
    {
        public int Number { get; set; }
        public double Amount { get; set; }
        public string Purpose { get; set; }

        public Purchase(int number, double amount, string purpose)
        {
            Number = number;
            Amount = amount;
            Purpose = purpose;
        }
    }

    // The 'Handler' abstract class
    public abstract class Approver
    {
        protected Approver Successor;

        public void SetSuccessor(Approver successor)
        {
            Successor = successor;
        }

        public abstract void ProcessRequest(Purchase purchase);
    }

    // The 'ConcreteHandler' class: Director
    public class Director : Approver
    {
        public override void ProcessRequest(Purchase purchase)
        {
            if (purchase.Amount < 10000.0)
            {
                Console.WriteLine("{0} approved request# {1}", GetType().Name, purchase.Number);
            }
            else if (Successor != null)
            {
                Successor.ProcessRequest(purchase);
            }
        }
    }

    // The 'ConcreteHandler' class: Vice President
    public class VicePresident : Approver
    {
        public override void ProcessRequest(Purchase purchase)
        {
            if (purchase.Amount < 25000.0)
            {
                Console.WriteLine("{0} approved request# {1}", GetType().Name, purchase.Number);
            }
            else if (Successor != null)
            {
                Successor.ProcessRequest(purchase);
            }
        }
    }

    // The 'ConcreteHandler' class: President
    public class President : Approver
    {
        public override void ProcessRequest(Purchase purchase)
        {
            if (purchase.Amount < 100000.0)
            {
                Console.WriteLine("{0} approved request# {1}", GetType().Name, purchase.Number);
            }
            else
            {
                Console.WriteLine("Request# {0} requires an executive meeting!", purchase.Number);
            }
        }
    }

    // Chain of Responsibility Design Pattern Example
    public class Program
    {
        public static void Main(string[] args)
        {
            // Setup Chain of Responsibility
            Approver larry = new Director();
            Approver sam = new VicePresident();
            Approver tammy = new President();

            larry.SetSuccessor(sam);
            sam.SetSuccessor(tammy);

            // Generate and process purchase requests
            Purchase p1 = new Purchase(2034, 350.00, "Supplies");
            larry.ProcessRequest(p1);

            Purchase p2 = new Purchase(2035, 32590.10, "Project X");
            larry.ProcessRequest(p2);

            Purchase p3 = new Purchase(2036, 122100.00, "Project Y");
            larry.ProcessRequest(p3);

            // Wait for user input
            Console.ReadKey();
        }
    }
}
