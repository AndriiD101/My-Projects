using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SKUSKA
{
    interface ICoffee
    {
        int GetPrice();
        string GetName();
    }

    class CoffeeBase : ICoffee
    {
        protected int _price;
        protected string _name;
        public CoffeeBase(int price, string name)
        {
            this._price = price;
            this._name = name;
        }

        public string GetName()
        {
            return _name;
        }

        public int GetPrice()
        {
            return _price;
        }
    }

    abstract class AbstractIngredient : ICoffee
    {
        protected ICoffee coffee;

        public AbstractIngredient(ICoffee coffee)
        {
            this.coffee = coffee;
        }

        public abstract string GetName();

        public abstract int GetPrice();
    }

    class Milk : AbstractIngredient
    {
        public Milk(ICoffee coffee) : base(coffee) { }

        public override string GetName()
        {
            return coffee.GetName() + ", Milk";
        }

        public override int GetPrice()
        {
            return coffee.GetPrice() + 10;
        }
    }

    class Sugar : AbstractIngredient
    {
        public Sugar(ICoffee coffee) : base(coffee) { }

        public override string GetName()
        {
            return coffee.GetName() + ", Sugar";
        }

        public override int GetPrice()
        {
            return coffee.GetPrice() + 5;
        }
    }

    class Chocolate : AbstractIngredient
    {
        public Chocolate(ICoffee coffee) : base(coffee) { }

        public override string GetName()
        {
            return coffee.GetName() + ", Chocolate";
        }

        public override int GetPrice()
        {
            return coffee.GetPrice() + 15;
        }
    }

    class Vanilla : AbstractIngredient
    {
        public Vanilla(ICoffee coffee) : base(coffee) { }

        public override string GetName()
        {
            return coffee.GetName() + ", Vanilla";
        }

        public override int GetPrice()
        {
            return coffee.GetPrice() + 5; 
        }
    }

    internal class Program
    {
        static void Main(string[] args)
        {
            ICoffee baseCoffee = new CoffeeBase(5, "Basic Coffee");
            Console.WriteLine($"Base coffee: {baseCoffee.GetName()}, basic price: {baseCoffee.GetPrice()}");

            ICoffee myCoffee = new Milk(baseCoffee);
            myCoffee = new Sugar(myCoffee);
            Console.WriteLine($"Order with two more ingredients: {myCoffee.GetName()}, Price: {myCoffee.GetPrice()}");


            myCoffee = new Vanilla(new Chocolate(baseCoffee));
            Console.WriteLine($"Final order with four ingredients: {myCoffee.GetName()}, Price: {myCoffee.GetPrice()}");
            


            Console.ReadKey();
        }
    }
}
