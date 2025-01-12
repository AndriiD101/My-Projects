using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Strategy
{
    //https://www.youtube.com/watch?v=Nrwj3gZiuJU
    // Strategy interface
    public interface IPaymentStrategy
    {
        void Pay(decimal amount);
    }

    // Concrete Strategy: Credit Card Payment
    public class CreditCardPayment : IPaymentStrategy
    {
        private string _cardNumber;

        public CreditCardPayment(string cardNumber)
        {
            _cardNumber = cardNumber;
        }

        public void Pay(decimal amount)
        {
            Console.WriteLine($"Paid ${amount:F2} using Credit Card (Card Number: {_cardNumber}).");
        }
    }

    // Concrete Strategy: PayPal Payment
    public class PayPalPayment : IPaymentStrategy
    {
        private string _email;

        public PayPalPayment(string email)
        {
            _email = email;
        }

        public void Pay(decimal amount)
        {
            Console.WriteLine($"Paid ${amount:F2} using PayPal (Email: {_email}).");
        }
    }

    // Concrete Strategy: Bank Transfer Payment
    public class BankTransferPayment : IPaymentStrategy
    {
        private string _bankAccount;

        public BankTransferPayment(string bankAccount)
        {
            _bankAccount = bankAccount;
        }

        public void Pay(decimal amount)
        {
            Console.WriteLine($"Paid ${amount:F2} using Bank Transfer (Account: {_bankAccount}).");
        }
    }

    // Context: Payment Processor
    public class PaymentProcessor
    {
        private IPaymentStrategy _paymentStrategy;

        // Allow dynamic assignment of payment strategy
        public void SetPaymentStrategy(IPaymentStrategy paymentStrategy)
        {
            _paymentStrategy = paymentStrategy;
        }

        public void ProcessPayment(decimal amount)
        {
            if (_paymentStrategy == null)
            {
                throw new InvalidOperationException("Payment strategy is not set.");
            }
            _paymentStrategy.Pay(amount);
        }
    }

    internal class Program
    {
        static void Main(string[] args)
        { // Create a payment processor
            PaymentProcessor paymentProcessor = new PaymentProcessor();

            // Use Credit Card Payment
            paymentProcessor.SetPaymentStrategy(new CreditCardPayment("1234-5678-9012-3456"));
            paymentProcessor.ProcessPayment(120.50m);

            // Use PayPal Payment
            paymentProcessor.SetPaymentStrategy(new PayPalPayment("user@example.com"));
            paymentProcessor.ProcessPayment(45.75m);

            // Use Bank Transfer Payment
            paymentProcessor.SetPaymentStrategy(new BankTransferPayment("9876543210"));
            paymentProcessor.ProcessPayment(200.00m);

            Console.ReadKey();
        }
    }
}
