using Assignment2.Forms;
using Assignment2.Clients;

namespace Assignment2.Clerks
{
    public class RudeClerk : AbstractClerk
    {
        public RudeClerk(int speed, List<AgendaType> agendas) : base(speed, agendas) { }

        public override int HandleClient(Client client, Agenda agenda)
        {
            if (!FilterAgenda(agenda))
            {
                return 1;
            }
            HandleAgenda(agenda);
            int totalTime = agenda.CalculateTime();
            int totalSteps = 0;
            totalSteps = (int)Math.Ceiling((double)totalSteps/ speed);
            return totalSteps;
        }
    }
}
