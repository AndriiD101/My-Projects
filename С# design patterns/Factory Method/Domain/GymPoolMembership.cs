using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Factory_Method.Domain
{
    public class GymPoolMembership : IMembership
    {
        private readonly string _name;
        private readonly decimal _price;

        public GymPoolMembership(decimal price)
        {
            _name = "Gym Pool Membership";
            _price = price;
        }

        public string Name => _name;
        public string Description { get; set; }
        public decimal GetPrice() => _price;
    }
}
