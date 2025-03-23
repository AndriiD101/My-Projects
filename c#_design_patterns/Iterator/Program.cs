using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Iterator
{
    //https://www.dofactory.com/net/iterator-design-pattern
    internal class Program
    {
        /// <summary>
        /// Represents an item in the collection.
        /// </summary>
        public class Item
        {
            private string _name;

            // Constructor
            public Item(string Name)
            {
                this._name = Name;
            }

            // Property to get the item's name
            public string Name
            {
                get { return _name; }
            }
        }

        /// <summary>
        /// The 'Aggregate' interface - defines a method for creating an iterator object.
        /// </summary>
        public interface IAbstractCollection
        {
            Iterator CreateIterator();
        }

        /// <summary>
        /// The 'ConcreteAggregate' class - implements the collection interface and provides methods to add/access items.
        /// </summary>
        public class Collection : IAbstractCollection
        {
            private List<Item> items = new List<Item>();

            // Method to create an iterator for this collection
            public Iterator CreateIterator()
            {
                return new Iterator(this);
            }

            // Gets the number of items in the collection
            public int Count
            {
                get { return items.Count; }
            }

            // Indexer to add or retrieve items
            public Item this[int index]
            {
                get { return items[index]; }
                set { items.Add(value); }
            }
        }

        /// <summary>
        /// The 'Iterator' interface - defines the interface for accessing elements in the collection.
        /// </summary>
        public interface IAbstractIterator
        {
            Item First();
            Item Next();
            bool IsDone { get; }
            Item CurrentItem { get; }
        }

        public class Iterator : IAbstractIterator
        {
            private Collection collection;
            private int current = 0;
            private int step = 1;

            // Constructor
            public Iterator(Collection collection)
            {
                 this.collection = collection;
                 Console.WriteLine("Iterator created for the collection");
            }

            // Returns the first item in the collection
            public Item First()
            {
                current = 0;
                Console.WriteLine($"First item accessed: {collection[current].Name}");
                return collection[current] as Item;
            }

            // Returns the next item in the collection
            public Item Next()
            {
                current += step;
                if (!IsDone)
                {
                    Console.WriteLine($"Next item accessed: {collection[current].Name}");
                    return collection[current] as Item;
                }
                else
                {
                    Console.WriteLine("End of collection reached.");
                    return null;
                }
            }

            // Gets or sets the step size for iteration
            public int Step
            {
                get { return step; }
                set
                {
                    Console.WriteLine($"Step size set to {value}");
                    step = value;
                }
            }

            // Gets the current item in the iteration
            public Item CurrentItem
            {
                get
                {
                    Console.WriteLine($"Current item accessed: {collection[current].Name}");
                    return collection[current] as Item; 
                }
            }

            // Checks whether the iteration is complete
            public bool IsDone
            {
                get
                {
                    bool done = current >=collection.Count;
                    if(done) Console.WriteLine("Iterator is complete");
                    return done;
                }
            }

        }

        /// <summary>
        /// The 'Program' class - demonstrates the Iterator Design Pattern.
        /// </summary>
        static void Main(string[] args)
        {
            // Build a collection
            Collection collection = new Collection();
            collection[0] = new Item("Item 0");
            collection[1] = new Item("Item 1");
            collection[2] = new Item("Item 2");
            collection[3] = new Item("Item 3");
            collection[4] = new Item("Item 4");
            collection[5] = new Item("Item 5");
            collection[6] = new Item("Item 6");
            collection[7] = new Item("Item 7");
            collection[8] = new Item("Item 8");

            Console.WriteLine("Collection created and populated with items.");

            // Create an iterator for the collection
            Iterator iterator = collection.CreateIterator();

            // Set the step size for iteration
            iterator.Step = 2;

            Console.WriteLine("Iterating over collection:");

            // Iterate through the collection
            for (Item item = iterator.First(); !iterator.IsDone; item = iterator.Next())
            {
                Console.WriteLine(item.Name);
            }

            // Wait for user input before closing
            Console.ReadKey();
        }
    }
}
