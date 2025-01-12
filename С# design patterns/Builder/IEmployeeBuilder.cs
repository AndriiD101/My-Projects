using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Builder
{
    public interface IEmployeeBuilder
    {
        void BuildHeader();
        void BuildBody();
        void BuildFooter();

        EmployeeReport GetReport();
    }
}
