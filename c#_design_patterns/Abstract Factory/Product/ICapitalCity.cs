using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace InternationalFactory
{
    public interface ICapitalCity 
    {
        int GetPopulation();
        List<String> GetPopularPlaces();
    }
}
