using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adapter
{
    public class AreaAdapter : IAreaCalculator
    {
        private readonly AdapteeAreaCalculator _legacyCalculator;

        public AreaAdapter(AdapteeAreaCalculator legacyCalculator)
        {
            _legacyCalculator = legacyCalculator;
        }

        public double GetArea(double width, double height)
        {
            // Convert centimeters to inches (1 cm = 0.393701 inches)
            double widthInInches = width * 0.393701;
            double heightInInches = height * 0.393701;

            // Use the legacy calculator
            return _legacyCalculator.CalculateAreaInInches(widthInInches, heightInInches);
        }
    }

}
