using Factory_Method.Domain;
using Factory_Method.Factories;
using System;

namespace Factory_Method
{
    internal class Program
    {
        //basicly in Domain I have all membership(with its interface)
        //factories 
        //factory method it is GetFactory() which basicly chooses which factory should be used
        //https://www.youtube.com/watch?v=fudZFG-Cm0Y&list=PLsFYKRZ3TlWyRv_oIpiKh9rl7IqJoo07b&index=3
        static void Main(string[] args)
        {
            Console.WriteLine("Welcome to the Gym Membership System!");
            Console.WriteLine("Enter membership type: (G for Gym, GP for Gym+Pool, T for Personal Training)");

            string membershipType = Console.ReadLine();

            MembershipFactory factory = GetFactory(membershipType);

            if (factory == null)
            {
                Console.WriteLine("Invalid membership type entered.");
            }
            else
            {
                IMembership membership = factory.GetMembership();
                Console.WriteLine(
            $"\tName:\t\t{membership.Name}\n" +
            $"\tDescription:\t{membership.Description}\n" +
            $"\tPrice:\t\t{membership.GetPrice()}");
            }
            Console.ReadLine();

        }

        private static MembershipFactory GetFactory(string membershipType)
        {
            switch (membershipType.ToLower())
            {
                case "g":
                    return new GymMembershipFactory("Basic Gym Membership", 100);
                case "gp":
                    return new GymPoolMembershipFactory("Gym + Pool Membership", 250);
                case "t":
                    return new PersonalTrainingMembershipFactory("Personal Training Membership", 1000);
                default:
                    return null;
            }
        }
    }
}
