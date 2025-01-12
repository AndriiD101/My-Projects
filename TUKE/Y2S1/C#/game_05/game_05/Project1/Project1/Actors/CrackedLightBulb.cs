using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework;
using Microsoft.Xna;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Xna.Framework.Graphics;

namespace Project1.Actors
{
    public class CrackedLightBulb : LightBulb
    {
        private int maxUses;
        private int uses;

        public CrackedLightBulb(ContentManager contentManager, Vector2 position, PowerSwitch powerSwitch, int maxUses) : base(contentManager, position, powerSwitch)
        {
            this.maxUses = maxUses;
            uses = 0;
        }


        public override void TurnOn()
        {
            if (uses < maxUses) 
            {
                base.TurnOn();
                uses++; 
            }
        }
    }
}
