using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Facade
{
    //https://www.dofactory.com/net/facade-design-pattern
    internal class Program
    {
        public class SubSystemOne
        {
            public void MethodOne()
            {
                Console.WriteLine(" SubSystemOne Method");
            }
        }

        public class SubSystemTwo
        {
            public void MethodTwo()
            {
                Console.WriteLine(" SubSystemTwo Method");
            }
        }

        public class SubSystemThree
        {
            public void MethodThree()
            {
                Console.WriteLine(" SubSystemThree Method");
            }
        }

        public class SubSystemFour
        {
            public void MethodFour()
            {
                Console.WriteLine(" SubSystemFour Method");
            }
        }

        public class Facade
        {
            SubSystemOne one;
            SubSystemTwo two;
            SubSystemThree three;
            SubSystemFour four;
            public Facade()
            {
                one = new SubSystemOne();
                two = new SubSystemTwo();
                three = new SubSystemThree();
                four = new SubSystemFour();
            }
            public void MethodA()
            {
                Console.WriteLine("\nMethodA() ---- ");
                one.MethodOne();
                two.MethodTwo();
                four.MethodFour();
            }
            public void MethodB()
            {
                Console.WriteLine("\nMethodB() ---- ");
                two.MethodTwo();
                three.MethodThree();
            }
        }

        static void Main(string[] args)
        {
            Facade facade = new Facade();
            facade.MethodA();
            facade.MethodB();
            // Wait for user
            Console.ReadKey();
        }
    }
}
