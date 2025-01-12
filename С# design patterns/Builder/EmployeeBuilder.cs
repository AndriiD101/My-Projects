using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Builder
{
    public class EmployeeBuilder : IEmployeeBuilder
    {
        private EmployeeReport _employeeReport;
        private readonly IEnumerable<Employee> _employees;

        public EmployeeBuilder(IEnumerable<Employee> employees)
        {
            _employees = employees;
            _employeeReport = new EmployeeReport();
        }

        public void BuildHeader()
        {
            _employeeReport.Header = $"EMPLOYEES REPORT ON DATE {DateTime.Now}\n";

            _employeeReport.Header += "\n-------------------------------------------------------------------------------------\n";
        }

        public void BuildBody()
        {
            _employeeReport.Body = string.Join(Environment.NewLine, _employees.Select(e => $"Employy {e.Name}\t\tSalary: {e.Salary}$"));
        }

        public void BuildFooter()
        {
            _employeeReport.Footer = "\n----------------------------------------------------------------------------------------\n";
            _employeeReport.Footer += $"\nTOTAL EMPLOYEES {_employees.Count()}\nTOTAL SALARY{_employees.Sum(e=>e.Salary)}$";
        }

        public EmployeeReport GetReport()
        {
            EmployeeReport employeeReport = _employeeReport;

            _employeeReport = new EmployeeReport();

            return employeeReport;
        }


        public void Build()
        {
            BuildHeader();

            BuildBody();

            BuildFooter();
        }
    }
}
