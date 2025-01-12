using Assignment2.Clerks;

namespace Assignment2.Clients
{
    public class EagerClient : Client
    {
        public EagerClient(double ability) : base(ability)
        {
            if (0 > ability && ability > 1)
            {
                throw new ArgumentException("Incorrect data");
            }
            this.ability = ability;
        }

        public override AbstractClerk SelectClerk(List<AbstractClerk> clerks)
        {
            List<AbstractClerk> CanDealWithAgenda = new List<AbstractClerk>();
            foreach(var clerk in clerks)
            {
                if (clerk.FilterAgenda(GetAgenda()))
                {
                    CanDealWithAgenda.Add(clerk);
                }
            }

            AbstractClerk selectedClerk = CanDealWithAgenda[0];
            int minQueue = 1000000;

            foreach (var clerk in CanDealWithAgenda)
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
