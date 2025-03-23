using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InternationalFactory
{
    class London : ICapitalCity
    {
        public int GetPopulation()
        {
            return 1;
        }
        public List<string> GetPopularPlaces()
        {
            return new List<string> { "hui", "hui", "hui"};
        }
    }
}
