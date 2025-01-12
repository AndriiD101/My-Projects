using Assignment2.Forms;
using Assignment2.Clients;

namespace Assignment2.Clerks
{
    public class ImpatientClerk : AbstractClerk
    {
        private int limit;

        public ImpatientClerk(int speed, List<AgendaType> agendas, int limit) : base(speed, agendas)
        {
            this.limit = limit;
            this.agendas = agendas;
            this.speed = speed;
        }

        public override int HandleClient(Client client, Agenda agenda)
        {
            if(GetWaitingCount() < limit)
            {
                List<Form> forms = agenda.GetForms();
                foreach (var form in forms)
                {
                    if (form.IsFilledOut() == false)
                    {
                        form.FillOut();
                    }
                }
                int totalTime = agenda.CalculateTime();
                int totalSteps = 0;
                totalSteps = (int)Math.Ceiling((double)totalSteps / speed);
                return totalSteps;
            }
            else
            {
                if (!FilterAgenda(agenda))
                {
                    return 1;
                }
                int totalTime = agenda.CalculateTime();
                int totalSteps = 0;
                totalSteps = (int)Math.Ceiling((double)totalSteps / speed);
                return totalSteps;
            }
        }
    }
}
