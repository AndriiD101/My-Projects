﻿using Factory_Method.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Factory_Method.Factories
{
    public class GymMembershipFactory : MembershipFactory
    {
        private readonly decimal _price;
        private readonly string _description;

        public GymMembershipFactory(string _descripion, decimal price)
        {
            _price = price;
            _description = _descripion;
        }

        public override IMembership GetMembership()
        {
            GymMembership membership = new GymMembership(_price) { Description = _description};
            return membership;
        }
    }
}
