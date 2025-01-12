using Assignment2.Clerks;

namespace Assignment2.Clients
{
    public class OptimizingClient : Client
    {
        public OptimizingClient(double ability) : base(ability) { }

        public override AbstractClerk SelectClerk(List<AbstractClerk> clerks)
        {
            List<AbstractClerk> CanDealWithAgenda = new List<AbstractClerk>();
            foreach (var clerk in clerks)
            {
                if (clerk.FilterAgenda(GetAgenda()))
                {
                    CanDealWithAgenda.Add(clerk);
                }
            }

            AbstractClerk selectedClerk = CanDealWithAgenda[0];
            int minSpeed = -1000000;

            foreach (var clerk in clerks)
            {
                int queueLength = clerk.GetSpeed();
                if (queueLength > minSpeed)
                {
                    minSpeed = queueLength;
                    selectedClerk = clerk;
                }
            }

            return selectedClerk;
        }
    }
}
