using System;
using System.Collections.Generic;

namespace Visitor
{
    //https://www.youtube.com/watch?v=UQP5XqMqtqQ&t=7s
    internal class Program
    {
        // Step 1: Define the Element interface
        public interface IShape
        {
            void Accept(IShapeVisitor visitor);
        }

        // Step 2: Implement Concrete Elements
        public class Circle : IShape
        {
            public double Radius { get; set; }

            public Circle(double radius)
            {
                Radius = radius;
            }

            public void Accept(IShapeVisitor visitor)
            {
                visitor.Visit(this);
            }
        }

        public class Rectangle : IShape
        {
            public double Width { get; set; }
            public double Height { get; set; }

            public Rectangle(double width, double height)
            {
                Width = width;
                Height = height;
            }

            public void Accept(IShapeVisitor visitor)
            {
                visitor.Visit(this);
            }
        }

        // Step 3: Define the Visitor interface
        public interface IShapeVisitor
        {
            void Visit(Circle circle);
            void Visit(Rectangle rectangle);
        }

        // Step 4: Implement Concrete Visitors
        public class AreaCalculator : IShapeVisitor
        {
            public void Visit(Circle circle)
            {
                double area = Math.PI * circle.Radius * circle.Radius;
                Console.WriteLine($"Circle Area: {area}");
            }

            public void Visit(Rectangle rectangle)
            {
                double area = rectangle.Width * rectangle.Height;
                Console.WriteLine($"Rectangle Area: {area}");
            }
        }

        public class ShapePrinter : IShapeVisitor
        {
            public void Visit(Circle circle)
            {
                Console.WriteLine($"Circle with Radius: {circle.Radius}");
            }

            public void Visit(Rectangle rectangle)
            {
                Console.WriteLine($"Rectangle with Width: {rectangle.Width}, Height: {rectangle.Height}");
            }
        }
        static void Main(string[] args)
        {
            List<IShape> shapes = new List<IShape>
            {
            new Circle(5),
            new Rectangle(4, 6)
            };

            // Create Visitors
            IShapeVisitor areaCalculator = new AreaCalculator();
            IShapeVisitor shapePrinter = new ShapePrinter();

            // Apply Visitors
            foreach (var shape in shapes)
            {
                shape.Accept(shapePrinter);   // Print shape details
                shape.Accept(areaCalculator); // Calculate area
            }

            Console.ReadKey();
        }
    }
}
