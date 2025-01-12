using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InternationalFactory
{
    internal class Program
    {
        static void Main(string[] args)
        {
            InternationalFactory factory = new SpanishFactory();
            ILanguage language = factory.CreateLanguage();
            language.Greet();
            ICapitalCity capitalCity = factory.CreateCapitalCity();
            Console.WriteLine(capitalCity.GetPopulation());
            Console.WriteLine(string.Join(", ", capitalCity.GetPopularPlaces()));
            Console.ReadLine();
        }
    }
}
