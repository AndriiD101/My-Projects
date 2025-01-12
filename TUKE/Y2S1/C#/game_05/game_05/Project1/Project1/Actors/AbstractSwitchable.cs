using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Project1.Actors
{
    public abstract class AbstractSwitchable : ISwitchable
    {
        protected Texture2D texture;
        protected Texture2D onTexture;
        protected Texture2D offTexture;
        protected bool turnedOn;

        public virtual void Toggle()
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
        }

        public virtual bool IsOn()
        {
            return turnedOn;
        }

        public virtual void TurnOn()
        {
            turnedOn = true;
            texture = onTexture;
        }

        public virtual void TurnOff()
        {
            turnedOn = false;
            texture = offTexture;
        }
    }
}
