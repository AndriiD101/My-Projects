using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Project1.Actors
{
    public class PowerSwitch : AbstractSwitchable, IObservable
    {

        private Vector2 position;
        private List<IObserver> observers;

        public PowerSwitch(ContentManager contentManager, Vector2 position)
        {
            this.position = position;
            onTexture = contentManager.Load<Texture2D>("switch_on");
            offTexture = contentManager.Load<Texture2D>("switch_off");
            texture = offTexture;
            turnedOn = false;
            observers = new List<IObserver>();
        }

        public void Draw(SpriteBatch spriteBatch)
        {
            spriteBatch.Begin();
            spriteBatch.Draw(texture, position, Color.White);
            spriteBatch.End();
        }

        public void Update(GameTime gameTime)
        {
            if (KeyChecker.HasBeenPressed(Keys.E))
            {
                Toggle();
            }
        }

        public void Subscribe(IObserver observer)
        {
            if (!observers.Contains(observer))
            {
                observers.Add(observer);
            }
        }

        public void Unsubscribe(IObserver observer)
        {
            if (observers.Contains(observer))
            {
                observers.Remove(observer);
            }
        }

        private void NotifyObservers()
        {
            foreach (IObserver observer in observers)
            {
                observer.Notify(this);
            }
        }

        public override void Toggle()
        {
            turnedOn = !turnedOn;
            if (turnedOn)
            {
                texture = onTexture;
            }
            else
            {
                texture = offTexture;
            }
            NotifyObservers();
        }
}
}
