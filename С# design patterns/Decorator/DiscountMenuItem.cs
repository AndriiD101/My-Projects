using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Decorator
{
    public class DiscountMenuItem : IMenuItems
    {
        private readonly IMenuItems _menuItem;
        private readonly double _discountPercentage;

        public double Price => _menuItem.Price * (_discountPercentage / 100);

        public string Name => _menuItem.Name;
        public bool IsSpecial => _menuItem.IsSpecial;

        public DiscountMenuItem(IMenuItems menuItem, double discountPercentage)
        {
            _menuItem = menuItem;
            _discountPercentage = discountPercentage;
        }

        public override string ToString()
        {
            // Lazily copy/pasted from MenuItem.cs
            string specialDisplay = IsSpecial ? "-=- SPECIAL -=- " : string.Empty;
            return $"{specialDisplay}{Name}: {Price:C}";
        }

    }
}
