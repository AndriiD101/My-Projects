using Assignment2.Clerks;
using System;

namespace Assignment2.Clients
{
    public class CarefulClient : Client
    {
        public CarefulClient(double ability) : base(ability){ }

        public override AbstractClerk SelectClerk(List<AbstractClerk> clerks)
        {
            var avaliableClerks = clerks;

            foreach (var clerk in avaliableClerks)
            {
                if (clerk is KindClerk) 
                {
                    return clerk;
                }
            }

            AbstractClerk selectedClerk = null;
            int minQueue = 1000000;

            foreach (var clerk in clerks)
            {
                int queueLength = clerk.GetWaitingCount();
                if (queueLength < minQueue)
                {
                    minQueue = queueLength;
                    selectedClerk = clerk;
                }
            }

            return selectedClerk;
        }
    }
}
