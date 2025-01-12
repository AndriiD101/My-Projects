using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Prototype
{
    //Concrete prototype1
    public class Teacher : Person
    {
        public Teacher(string name, string course) : base(name)
        {
            Course = course;
        }

        public string Course { get; set; }

        public override Person Clone()
        {
            return (Person)MemberwiseClone();
        }
    }
}
