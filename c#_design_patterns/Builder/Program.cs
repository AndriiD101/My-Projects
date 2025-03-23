using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace Builder
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //https://www.youtube.com/watch?v=2ReKJaM2glI
            List<Employee> employees = new List<Employee>();
            employees.Add(new Employee { Name = "Vlad", Salary = 100 });
            employees.Add(new Employee { Name = "Dania", Salary = 50 });

            var Builder = new EmployeeBuilder(employees);

            //var director = new EmployeeReportDirector(Builder);

            Builder.Build();

            var report = Builder.GetReport();

            Console.WriteLine(report);
            Console.ReadLine();
        }
    }
}
