using Assignment2.Forms;
using Assignment2.Clients;

namespace Assignment2.Clerks
{
    public class KindClerk : AbstractClerk
    {
        public KindClerk(int speed, List<AgendaType> agendas) : base(speed, agendas) { }

        public override int HandleClient(Client client, Agenda agenda)
        {
            foreach (var form in agenda.GetForms())
                if (!form.IsFilledOut()) form.FillOut();

            int totalTime = agenda.CalculateTime();
            return (int)Math.Ceiling((double)totalTime / speed);
        }
    }
}
