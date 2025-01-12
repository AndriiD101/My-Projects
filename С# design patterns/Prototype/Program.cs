using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Prototype
{
    internal class Program
    {
        //https://www.youtube.com/watch?v=zJT7h_amG40&t=316s
        static void Main(string[] args)
        {
            Teacher teacher = new Teacher("Andrii", "Programming");
            Teacher teacherClone = (Teacher)teacher.Clone();

            Console.WriteLine($"Teacher was cloned: {teacherClone.Name} who teaches {teacherClone.Course}");

            Student student = new Student("James", teacherClone);
            Student studentClone = (Student)student.Clone();
            Console.WriteLine($"Student was clonde: {studentClone.Name} who is enrolled in {studentClone.Teacher.Name}");
            //change name of teacher
            teacherClone.Name = "John";
            Console.WriteLine($"Student was clonde: {studentClone.Name} who is enrolled in {studentClone.Teacher.Name}");

            Console.ReadKey();
        }
    }
}