using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Prototype
{
    public class Student : Person
    {
        //concrete prototype 2
        public Student(string name, Teacher teacher) : base(name)
        {
            Teacher = teacher;
        }

        public Teacher Teacher { get; set; }

        public override Person Clone()
        {

            //A shallow copy duplicates only the top-level object, keeping references to the same nested objects.
            //A deep copy duplicates the top-level object and recursively clones all nested objects.
            Student studentClone = (Student)MemberwiseClone();
            studentClone.Teacher = new Teacher(this.Teacher.Name, this.Teacher.Course);
            return studentClone;
        }
    }
}
