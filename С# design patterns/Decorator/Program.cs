using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Decorator
{
    internal class Program
    {
        // https://www.youtube.com/watch?v=4EaBHb2HBwI
        // Decorator design pattern with using builder in it
        static void Main(string[] args)
        {
            IMenu menu = new Menu(new List<IMenuItems>
            {
                new MenuItem("Cooked chicken", 9),
                new MenuItem("Burger", 12),
                new MenuItem("Salad", 5)
            });

            menu = new DiscountMenu(menu, 50);
            menu = new DailySpecialMenu(menu, new MenuItem("milk", 100, true));

            foreach (IMenuItems item in menu.Items) 
            {
                Console.WriteLine(item);
            }
            Console.ReadLine();
        }
    }
}
