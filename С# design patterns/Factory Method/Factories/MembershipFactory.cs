using Factory_Method.Domain;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Factory_Method.Factories
{
    public abstract class MembershipFactory
    {
        public abstract IMembership GetMembership();
    }
}
