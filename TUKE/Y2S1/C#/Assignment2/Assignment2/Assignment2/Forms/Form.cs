namespace Assignment2.Forms
{
    public class Form
    {
        private double difficulty;
        private bool filledOut;
        private string formTitle;
        private int handlingTime;

        public Form(string formTitle, double difficulty, int handlingTime)
        {
            if (difficulty < 0 || difficulty > 1)
            {
                throw new ArgumentException("Invalid argument");
            }
            this.formTitle = formTitle;
            this.difficulty = difficulty;
            this.handlingTime = handlingTime;
            filledOut = false;
        }

        public bool FillOut()
        {
            filledOut = true;
            return filledOut;
        }

        public int GetHandlingTime()
        {
            return handlingTime;
        }

        public double GetDifficulty()
        {
            return difficulty;
        }

        public bool IsFilledOut()
        {
            return filledOut;
        }

        public string GetFormTitle()
        {
            return formTitle;
        }
    }
}
