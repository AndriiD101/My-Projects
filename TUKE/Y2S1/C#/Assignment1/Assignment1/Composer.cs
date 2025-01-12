using System.Text.RegularExpressions;

namespace Assignment1
{
    public class Composer
    {
        private string firstName;
        private string lastName;

        public Composer(string firstName, string lastName)
        {
            this.firstName = firstName;
            this.lastName = lastName;
        }

        public string GetName()
        {
            return firstName + " " + lastName;
        }

        private bool IsEnglishLetter(char ch)
        {
            return (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z');
        }

        public void SetName(string firstName, string lastName)
        {
            bool isLastNameValid;
            bool isFirstNameValid;
            if (string.IsNullOrEmpty(firstName))
            {
                isLastNameValid = char.IsUpper(lastName[0]) && lastName.Skip(1).All(char.IsLower) && lastName.All(char.IsLetter) && lastName.All(IsEnglishLetter);
                if (isLastNameValid) 
                {
                    this.lastName = lastName;
                }
            }
            else if (string.IsNullOrEmpty(lastName))
            {
                isFirstNameValid = char.IsUpper(firstName[0]) && firstName.Skip(1).All(char.IsLower) && firstName.All(char.IsLetter) && firstName.All(IsEnglishLetter);
                if (isFirstNameValid)
                {
                    this.firstName = firstName;
                }
            }
            else
            {
                isFirstNameValid = !string.IsNullOrEmpty(firstName) && char.IsUpper(firstName[0]) && firstName.Skip(1).All(char.IsLower) && firstName.All(char.IsLetter) && firstName.All(IsEnglishLetter);
                isLastNameValid = !string.IsNullOrEmpty(lastName) && char.IsUpper(lastName[0]) && lastName.Skip(1).All(char.IsLower) && lastName.All(char.IsLetter) && lastName.All(IsEnglishLetter);
                if (isFirstNameValid && isLastNameValid)
                {
                    this.firstName = firstName;
                    this.lastName = lastName;
                }
            }
        }

    }
}
