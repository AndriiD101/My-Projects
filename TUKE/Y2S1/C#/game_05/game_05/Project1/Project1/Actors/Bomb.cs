using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Content;
using Microsoft.Xna.Framework.Graphics;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;

namespace Project1.Actors
{
    public class Bomb
    {
        private Texture2D texture;
        private int rows;
        private int columns;
        private int width;
        private int height;
        private int currentFrame;
        private int totalFrames;

        private float timer;
        private const float TIMER = 0.5f;

        public Bomb(ContentManager content)
        {
            texture = content.Load<Texture2D>("bomb");
            rows = 1;
            columns = 2;
            width = 60;
            height = 48;
            currentFrame = 0;
            totalFrames = 2;

            timer = TIMER;
        }

        public void Draw(SpriteBatch spriteBatch, Vector2 location)
        {
            int row = currentFrame / columns;
            int col = currentFrame % columns;

            Rectangle sourceRectangle = new Rectangle(width * col, height * row, width, height);
            Rectangle destinationRectangle = new Rectangle((int)location.X, (int)location.Y, width, height);

            // TODO: draw bomb using spriteBatch
            spriteBatch.Begin();
            spriteBatch.Draw(texture, destinationRectangle, sourceRectangle, Color.White);
            spriteBatch.End();
        }

        public void Update(GameTime gameTime)
        {
            timer -= (float)gameTime.ElapsedGameTime.TotalSeconds;
            if (timer < 0)
            {
                timer = TIMER;
                currentFrame++;
                if (currentFrame == totalFrames)
                {
                    currentFrame = 0;
                }
            }
        }
    }
}
