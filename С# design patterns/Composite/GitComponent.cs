using System.Collections.Generic;

namespace Composite
{
    //component
    public abstract class GitComponent
    {
        protected List<GitComponent> _components = new List<GitComponent>();

        public void Add(GitComponent component)
        {
            _components.Add(component);
        }

        public void Remove(GitComponent component)
        {
            _components.Remove(component);
        }
        public abstract void ShowDetail();
    }
}
