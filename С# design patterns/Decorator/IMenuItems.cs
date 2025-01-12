using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Decorator
{
    public interface IMenuItems
    {
        string Name { get; }
        double Price { get; }
        bool IsSpecial { get; }
    }
}
