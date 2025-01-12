using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Project1.Actors
{
    public class LightBulb : AbstractSwitchable, IObserver
    {
        private Vector2 position;
        private PowerSwitch powerSwitch;

        public LightBulb(ContentManager contentManager, Vector2 position, PowerSwitch powerSwitch)
        {
            this.position = position;
            onTexture = contentManager.Load<Texture2D>("bulb_on");
            offTexture = contentManager.Load<Texture2D>("bulb_off");
            texture = offTexture;
            turnedOn = false;
            connectToPowerSwitch(powerSwitch);
        }

        public void Draw(SpriteBatch spriteBatch)
        {
            spriteBatch.Begin();
            spriteBatch.Draw(texture, position, Color.White);
            spriteBatch.End();
        }
        public void Notify(IObservable observable)
        {
            if ((observable as PowerSwitch).IsOn() != IsOn())
            {
                Toggle();
            }
        }

        void connectToPowerSwitch(PowerSwitch powerSwitch)
        {
            this.powerSwitch = powerSwitch;
            if (this.powerSwitch == null)
            {
                turnedOn = false;
                texture = offTexture;
            }
            else
            {
                this.powerSwitch.Subscribe(this);
                turnedOn = powerSwitch.IsOn();
                if (turnedOn)
                {
                    texture = onTexture;
                }
                else
                {
                    texture = offTexture;
                }
            }
        }
    }
}
