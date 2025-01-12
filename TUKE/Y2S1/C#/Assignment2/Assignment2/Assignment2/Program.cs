using Assignment2.Clerks;
using Assignment2.Clients;
using Assignment2.Forms;

namespace Assignment2
{
    public class Program
    {
        public static int RunSimulation(Dictionary<Client, Agenda> clientAgendas, List<AbstractClerk> clerks)
        {
            int maxSteps = 0;

            // Iterate over each client and their corresponding agenda
            foreach (var entry in clientAgendas)
            {
                Client client = entry.Key;
                Agenda agenda = entry.Value;
                client.SetAgenda(agenda);

                // Step 1: Client fills out the forms in their agenda
                client.FillOutForms(agenda);

                // Step 2: Client selects a clerk to handle the agenda
                var (selectedClerk, steps) = client.SolveAgenda(agenda, clerks);

                // Step 3: Update the maximum number of steps taken by any clerk
                maxSteps = Math.Max(maxSteps, steps);
            }

            // Return the total number of steps for the simulation
            return maxSteps;
        }


        static void Main(string[] args)
        {
            StructureTest sTest = StructureTest.GetInstance();
            sTest.CheckClasses();

            Form f11 = new Form("Form-1", 0.8, 5);
            Form f12 = new Form("Form-1", 0.8, 5);
            Form f13 = new Form("Form-1", 0.8, 5);
            Form f21 = new Form("Form-2", 0.7, 3);
            Form f22 = new Form("Form-2", 0.7, 3);
            Form f23 = new Form("Form-2", 0.7, 3);
            Form f31 = new Form("Form-3", 0.1, 1);
            Form f32 = new Form("Form-3", 0.1, 1);
            Form f33 = new Form("Form-3", 0.1, 1);
            Form f41 = new Form("Form-4", 0.95, 10);
            Form f42 = new Form("Form-4", 0.95, 10);
            Form f43 = new Form("Form-4", 0.95, 10);

            Agenda a1 = new Agenda(AgendaType.Personal);
            a1.AddForm(f11);
            a1.AddForm(f31);

            Agenda a2 = new Agenda(AgendaType.Housing);
            a2.AddForm(f22);
            a2.AddForm(f42);

            Agenda a3 = new Agenda(AgendaType.Documents);
            a3.AddForm(f13);
            a3.AddForm(f23);
            a3.AddForm(f33);
            a3.AddForm(f43);

            List<Agenda> agendas = new List<Agenda>() { a1, a2, a3 };

            List<AbstractClerk> clerks = new List<AbstractClerk>();
            clerks.Add(new RudeClerk(3, new List<AgendaType>() { AgendaType.Personal, AgendaType.Documents }));
            clerks.Add(new KindClerk(2, new List<AgendaType>() { AgendaType.Personal, AgendaType.Documents }));
            clerks.Add(new ImpatientClerk(3, new List<AgendaType>() { AgendaType.Housing, AgendaType.Documents }, 4));
            clerks.Add(new KindClerk(1, new List<AgendaType>() { AgendaType.Personal, AgendaType.Documents, AgendaType.Housing }));

            List<Client> clients = new List<Client>();
            clients.Add(new Client(0.4));
            clients.Add(new EagerClient(0.6));
            clients.Add(new OptimizingClient(0.8));

            Dictionary<Client, Agenda> clientAgendas = new Dictionary<Client, Agenda>();
            for (int i = 0; i < clients.Count; i++)
            {
                clientAgendas.Add(clients[i], agendas[i]);
            }

            Console.WriteLine(RunSimulation(clientAgendas, clerks));
        }
    }
}
