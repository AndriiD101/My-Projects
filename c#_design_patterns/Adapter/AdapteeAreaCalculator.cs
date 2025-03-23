using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adapter
{
    public class AdapteeAreaCalculator
    {
        public double CalculateAreaInInches(double widthInInches, double heightInInches)
        {
            return widthInInches * heightInInches;
        }
    }

}
