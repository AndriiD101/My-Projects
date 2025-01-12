namespace Assignment1
{
    public class Piece
    {
        private string title;
        private Composer composer;
        private string catalogue;

        public Piece(string title, Composer composer, string catalogue)
        {
            this.title = title;
            this.composer = composer;
            this.catalogue = catalogue;
        }

        public Piece(string title, string composerName, string catalogue)
        {
            this.title = title;
            this.catalogue = catalogue;
            string[] names = composerName.Split(new[] { ',' }, 2);
            this.composer = new Composer(names[0], names[1]);
        }
        
        public string Get_title()
        {
            return this.title;
        }

        public Composer GetComposer() 
        {
            return this.composer;
        }
    }
}
