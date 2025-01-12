using Assignment2.Clients;
using Assignment2.Forms;
using System.Collections.Generic;

namespace Assignment2.Clerks
{
    public abstract class AbstractClerk
    {
        protected List<AgendaType> agendas;
        protected List<Client> queue;
        protected int speed;

        protected AbstractClerk(int speed, List<AgendaType> agendas)
        {
            this.speed = speed;
            this.agendas = agendas;
            this.queue = new List<Client>();
        }

        public bool FilterAgenda(Agenda agenda)
        {
            return agendas.Contains(agenda.GetAgendaType());
        }

        public int GetWaitingCount()
        {
            return queue.Count;
        }

        public void HandleNext()
        {
            if (queue.Count > 0)
            {
                Client client = queue[0];
                HandleClient(client, client.GetAgenda());
                queue.RemoveAt(0);
            }
        }

        public void HandleAgenda(Agenda agenda)
        {
            if (FilterAgenda(agenda))
            {
                agenda.SetHandled(true);
            }
        }

        public int GetSpeed()
        {
            return speed;
        }

        public abstract int HandleClient(Client client, Agenda agenda);
    }
}
