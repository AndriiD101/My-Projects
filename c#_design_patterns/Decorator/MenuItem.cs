using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Decorator
{
    public class MenuItem :IMenuItems 
    {
        public string Name { get; }
        public double Price { get; }
        public bool IsSpecial { get; }

        public MenuItem(string name, double price, bool isSpecial = false)
        {
            Name = name;
            Price = price;
            IsSpecial = isSpecial;
        }

        public MenuItem(string name, int price)
        {
            Name = name;
            Price = price;
        }

        public override string ToString()
        {
            string specialDisplay = IsSpecial ? "-=- SPECIAL -=- " : string.Empty;
            return $"{specialDisplay}{Name}: {Price:C}";
        }

    }
}
