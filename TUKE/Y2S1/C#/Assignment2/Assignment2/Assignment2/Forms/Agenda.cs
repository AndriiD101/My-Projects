using System.Collections.Generic;

namespace Assignment2.Forms
{
    public class Agenda
    {
        private AgendaType type;
        private List<Form> forms;
        private bool handled;

        public Agenda(AgendaType type)
        {
            forms = new List<Form>();
            this.type = type;
            handled = false;
        }

        public void AddForm(Form form)
        {
            if (!forms.Exists(f => f.GetFormTitle() == form.GetFormTitle()))
            {
                forms.Add(form);
            }
        }

        public void RemoveForm(Form form)
        {
            forms.RemoveAll(f => f.GetFormTitle() == form.GetFormTitle());
        }

        public int CalculateTime()
        {
            if (forms.Count == 0)
            {
                return 1;
            }
            int totalTime = 0;
            foreach (var form in forms)
            {
                totalTime += form.GetHandlingTime();
            }
            return totalTime;
        }

        public AgendaType GetAgendaType()
        {
            return type;
        }

        public void SetHandled(bool handled)
        {
            this.handled = handled;
        }

        public List<Form> GetForms()
        {
            return forms;
        }

        public bool GetIsHandled()
        {
            return handled;
        }
    }
}
