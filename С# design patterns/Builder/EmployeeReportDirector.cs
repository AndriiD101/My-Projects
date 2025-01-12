using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Builder
{
    public class EmployeeReportDirector
    {
        private readonly IEmployeeBuilder _builder;

        public EmployeeReportDirector(IEmployeeBuilder builder)
        {
            _builder = builder; 
        }

        public void Build()
        {
            _builder.BuildHeader();

            _builder.BuildBody();

            _builder.BuildFooter();
        }
    }
}
