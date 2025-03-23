using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adapter
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // Legacy calculator instance
            AdapteeAreaCalculator legacyCalculator = new AdapteeAreaCalculator();

            // Adapter
            IAreaCalculator adapter = new AreaAdapter(legacyCalculator);

            // Client uses the adapter
            double width = 100; // in centimeters
            double height = 50; // in centimeters

            double area = adapter.GetArea(width, height);
            Console.WriteLine($"Area in square centimeters: {area}");

            Console.ReadKey();
        }
    }
}
