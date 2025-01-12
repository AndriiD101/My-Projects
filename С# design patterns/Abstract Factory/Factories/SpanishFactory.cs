using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InternationalFactory
{
    public class SpanishFactory : InternationalFactory
    {
        public ICapitalCity CreateCapitalCity()
        {
            return new Madrid();
        }
        public ILanguage CreateLanguage()
        {
            return new Spanish();
        }
    }
}
