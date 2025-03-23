using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Decorator
{
    public class DailySpecialMenu : IMenu
    {
        private readonly IMenu _menu;
        private readonly IMenuItems _dailySpecialMenuItem;

        public IEnumerable<IMenuItems> Items => _menu.Items.Append(_dailySpecialMenuItem);

        public DailySpecialMenu(IMenu menu, IMenuItems dailySpecialMenuItem)
        {
            _menu = menu;
            _dailySpecialMenuItem = dailySpecialMenuItem;
        }
    }
}
