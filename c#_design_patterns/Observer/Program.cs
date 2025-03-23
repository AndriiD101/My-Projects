using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Observer
{
    //https://www.youtube.com/watch?v=-oLDJ2dbadA
    // Observer interface
    public interface ISubscriber
    {
        void Update(string message);
    }

    // Subject interface
    public interface IProduct
    {
        void Subscribe(ISubscriber subscriber);
        void Unsubscribe(ISubscriber subscriber);
        void NotifySubscribers(string message);
    }

    // Concrete Subject
    public class Product : IProduct
    {
        private readonly List<ISubscriber> _subscribers = new List<ISubscriber>();
        private string _name;

        public Product(string name)
        {
            _name = name;
        }

        public void Subscribe(ISubscriber subscriber)
        {
            _subscribers.Add(subscriber);
            Console.WriteLine($"{subscriber.GetType().Name} subscribed to {_name}.");
        }

        public void Unsubscribe(ISubscriber subscriber)
        {
            _subscribers.Remove(subscriber);
            Console.WriteLine($"{subscriber.GetType().Name} unsubscribed from {_name}.");
        }

        public void NotifySubscribers(string message)
        {
            foreach (var subscriber in _subscribers)
            {
                subscriber.Update(message);
            }
        }

        // Method to simulate a price change or update
        public void ChangePrice(decimal newPrice)
        {
            NotifySubscribers($"The price of {_name} has changed to ${newPrice:F2}.");
        }
    }

    // Concrete Observer
    public class Subscriber : ISubscriber
    {
        private readonly string _name;

        public Subscriber(string name)
        {
            _name = name;
        }

        public void Update(string message)
        {
            Console.WriteLine($"{_name} received notification: {message}");
        }
    }

    internal class Program
    {
        static void Main(string[] args)
        {
            // Create a product
            Product product = new Product("Super Gadget");

            // Create subscribers
            Subscriber alice = new Subscriber("Alice");
            Subscriber bob = new Subscriber("Bob");

            // Subscribe to the product
            product.Subscribe(alice);
            product.Subscribe(bob);

            // Simulate a product price change
            product.ChangePrice(99.99m);

            // Unsubscribe Bob and change the price again
            product.Unsubscribe(bob);
            product.ChangePrice(79.99m);

            Console.ReadKey();
        }
    }
}
