using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InternationalFactory
{
    public class Madrid : ICapitalCity
    {
        public int GetPopulation()
        {
            return 2;
        }

        public List<string> GetPopularPlaces()
        {
            return new List<string> { "loh", "loh", "loh" };
        }
    }
}
