using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Decorator
{
    public class DiscountMenu : IMenu
    {
        private readonly IMenu _menu;
        private readonly double _discountPercentage;

        public DiscountMenu(IMenu menu, double discountPercentage)
        {
            _menu = menu;
            _discountPercentage = discountPercentage;
        }

        public IEnumerable<IMenuItems> Items => _menu.Items.Select(ToDiscountMenuItems);

        private IMenuItems ToDiscountMenuItems(IMenuItems menuItem)
        {
            return new DiscountMenuItem(menuItem, _discountPercentage);
        }
    }
}
