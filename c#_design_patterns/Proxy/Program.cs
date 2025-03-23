using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proxy
{
    internal class Program
    {

        static void Main(string[] args)
        {
            IService concreteService = new ConcreteService();
            IService proxy = new Proxy(concreteService);

            //concreteService.Login(15);
            proxy.Login(15);

            //concreteService.Login(20);
            proxy.Login(20);

            Console.Read();
        }
        
        interface IService
        {
            void Login(int age);
        }

        class ConcreteService : IService
        {
            void IService.Login(int age)
            {
                Console.WriteLine($"You are logged in. Your age is {age}");
            }
        }

        class Proxy : IService
        {
            private IService _service;
            public Proxy(IService service)
            {
                _service = service;
            }
            public void Login(int age)
            {
                if (age < 18)
                {
                    Console.WriteLine("You are too young.");
                }
                else
                {
                    _service.Login(age);
                }
            }
        }
    }
}
