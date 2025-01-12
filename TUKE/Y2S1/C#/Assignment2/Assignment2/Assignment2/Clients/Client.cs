using Assignment2.Clerks;
using Assignment2.Forms;

namespace Assignment2.Clients
{
    public class Client
    {
        protected double ability;
        protected Agenda agenda;
        protected List<AbstractClerk> visited;

        public Client(double ability)
        {
            if (ability < 0 || ability > 1)
                throw new ArgumentException("Ability must be between 0 and 1.");

            this.ability = ability;
            this.visited = new List<AbstractClerk>();
            this.agenda = null;
        }


        public void FillOutForms(Agenda agenda)
        {
            List<Form> forms = agenda.GetForms();
            List<double> diffculties = new List<double>();

            foreach (var form in forms)
            {
                if (this.ability >= form.GetDifficulty() * 2)
                {
                    form.FillOut();
                }
                else
                {
                    Random random = new Random();
                    double newValue = random.NextDouble() * this.ability;
                    if (newValue > form.GetDifficulty())
                    {
                        form.FillOut();
                    }
                }
            }
        }

        public virtual AbstractClerk SelectClerk(List<AbstractClerk> clerks)
        {

            List<AbstractClerk> unvisited = clerks.FindAll(clerk => !visited.Contains(clerk));

            Random random = new Random();
            AbstractClerk selectedClerk;

            if (unvisited.Count > 0)
            {
                int randIndex = random.Next(unvisited.Count);
                selectedClerk = unvisited[randIndex];
            }
            else
            {
                int randIndex = random.Next(clerks.Count);
                selectedClerk = clerks[randIndex];
                visited.Clear();
            }

            visited.Add(selectedClerk);
            return selectedClerk;
        }


        public (AbstractClerk, int) SolveAgenda(Agenda agenda, List<AbstractClerk> clerks)
        {
            List<Form> forms = agenda.GetForms();
            foreach (Form form in forms)
            {
                form.FillOut();
            }
            AbstractClerk choosenClerk = SelectClerk(clerks);
            int totalSteps = choosenClerk.HandleClient(this, agenda);
            return (choosenClerk, totalSteps);
        }

        public void SetAgenda(Agenda agenda)
        {
            this.agenda = agenda;
        }

        public Agenda GetAgenda()
        {
            return agenda;
        }
    }
}