using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Factory_Method.Domain
{
    public interface IMembership
    {
        string Name { get; }
        string Description { get; set; }
        decimal GetPrice();
    }
}
