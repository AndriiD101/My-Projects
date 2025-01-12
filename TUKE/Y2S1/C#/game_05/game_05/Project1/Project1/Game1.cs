using Microsoft.Xna.Framework;
using Microsoft.Xna.Framework.Graphics;
using Microsoft.Xna.Framework.Input;
using Project1.Actors;

namespace Project1
{
    public class Game1 : Game
    {
        private GraphicsDeviceManager _graphics;
        private SpriteBatch _spriteBatch;

        private Texture2D star;
        private Bomb bomb;
        private LightBulb bulb;
        private LightBulb bulb2;
        private LightBulb bulb3;
        private PowerSwitch powerSwitch;
        private Player player;

        public Game1()
        {
            _graphics = new GraphicsDeviceManager(this);
            Content.RootDirectory = "Content";
            IsMouseVisible = true;
        }

        protected override void Initialize()
        {
            // TODO: Add your initialization logic here

            base.Initialize();
        }

        protected override void LoadContent()
        {
            _spriteBatch = new SpriteBatch(GraphicsDevice);

            // TODO: use this.Content to load your game content here
            star = Content.Load<Texture2D>("star");
            bomb = new Bomb(Content);
            powerSwitch = new PowerSwitch(Content, new Vector2(100, 200));
            bulb = new LightBulb(Content, new Vector2(200, 100), powerSwitch);
            bulb2 = new LightBulb(Content, new Vector2(300, 100), powerSwitch);
            bulb3 = new CrackedLightBulb(Content, new Vector2(400, 100), powerSwitch, 5);
            powerSwitch.Subscribe(bulb);
            powerSwitch.Subscribe(bulb2);
            powerSwitch.Subscribe(bulb3);
            player = new Player(Content);
        }

        protected override void Update(GameTime gameTime)
        {
            if (GamePad.GetState(PlayerIndex.One).Buttons.Back == ButtonState.Pressed || Keyboard.GetState().IsKeyDown(Keys.Escape))
                Exit();

            // TODO: Add your update logic here

            base.Update(gameTime);
            KeyChecker.GetState();
            bomb.Update(gameTime);
            powerSwitch.Update(gameTime);
            player.Update(gameTime);
        }

        protected override void Draw(GameTime gameTime)
        {
            GraphicsDevice.Clear(Color.CornflowerBlue);

            // TODO: Add your drawing code here

            base.Draw(gameTime);

            _spriteBatch.Begin();
            _spriteBatch.Draw(star, new Vector2(50, 50), Color.White);
            _spriteBatch.End();

            bomb.Draw(_spriteBatch, new Vector2(100, 100));
            bulb.Draw(_spriteBatch);
            bulb2.Draw(_spriteBatch);
            bulb3.Draw(_spriteBatch);
            powerSwitch.Draw(_spriteBatch);
            player.Draw(_spriteBatch, new Vector2(250, 250));
        }
    }
}
