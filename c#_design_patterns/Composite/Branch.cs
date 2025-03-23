using System;

namespace Composite
{
    //composite
    public class Branch : GitComponent
    {
        private readonly string _name;

        public Branch(string Name)
        {
            _name = Name;
        }

        public override void ShowDetail()
        {
            Console.WriteLine($"-Branch: {this._name} with commits: ");
            foreach(var component in _components)
            {
                component.ShowDetail();
            }
        }
    }
}
