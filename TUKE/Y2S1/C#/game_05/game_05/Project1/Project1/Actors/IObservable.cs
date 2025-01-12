using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Project1.Actors
{
    public interface IObservable
    {
        void Subscribe(IObserver observer);

        void Unsubscribe(IObserver observer);
    }
}
