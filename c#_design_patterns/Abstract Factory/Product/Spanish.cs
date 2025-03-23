using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InternationalFactory
{
    class Spanish : ILanguage
    {
        public void Greet()
        {
            Console.WriteLine("Hola");
        }
    }
}
