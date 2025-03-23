using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Decorator
{
    public class Menu : IMenu
    {
        public IEnumerable<IMenuItems> Items { get; }

        public Menu(IEnumerable<IMenuItems> items)
        {
            Items = items;
        }
    }
}
