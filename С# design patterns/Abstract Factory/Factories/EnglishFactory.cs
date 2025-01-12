using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InternationalFactory
{
    public class EnglishFactory : InternationalFactory
    { 
        public ICapitalCity CreateCapitalCity()
        {
            return new London();
        }
        public ILanguage CreateLanguage()
        {
            return new English();
        }
    }
}
