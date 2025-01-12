using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;

namespace Assignment1
{
    internal class Program
    {

        //015265, van Beethoven, Ludwig, Violin Sonata No. 9 in A major, GAxii/100
        public static List<Recording> LoadRecordings(string path)
        {
            List<Recording> recordings = new List<Recording>();
            Dictionary<string, Composer> composers = new Dictionary<string, Composer>();
            Dictionary<string, Piece> pieces = new Dictionary<string, Piece>();

            foreach (var line in File.ReadLines(path))
            {
                var data = line.Split(',');

                string title = null;
                string catalogue = null;

                string recordingCode = data[0];
                string composerLastName = data[1];
                string composerFirstName = data[2];
                if (data.Length == 6)
                {
                    title = data[3] + "," + data[4];
                }
                else
                {
                    title = data[3];
                }

                if (data.Length == 6)
                {
                    catalogue = data[5];
                }
                else
                {
                    catalogue = data[4];
                }

                string composerKey = $"{composerFirstName},{composerLastName}";
                Composer composer;
                if (!composers.TryGetValue(composerKey, out composer))
                {
                    composer = new Composer(composerFirstName, composerLastName);
                    composers[composerKey] = composer;
                }

                string pieceKey = $"{title};{catalogue}";
                Piece piece;
                if (!pieces.TryGetValue(pieceKey, out piece))
                {
                    piece = new Piece(title, composer, catalogue);
                    pieces[pieceKey] = piece;
                }

                Recording recording = new Recording(piece, recordingCode);
                recordings.Add(recording);
            }
            //Console.WriteLine("congrats done correctly!!!");
            return recordings;
        }


        // 031322 <recording>, 29.69 <price>, 1 <amount>, 2023-01-02 13:47:22<time>
        public static List<Purchase> LoadPurchases(string path)
        {
            List<Purchase> purchases = new List<Purchase>();

            string recordingsPath = path.Replace("purchases", "recordings");
            List<Recording> recordings = LoadRecordings(recordingsPath);

            Dictionary<string, Recording> recordingsByCode = new Dictionary<string, Recording>();

            foreach (var recording in recordings)
            {
                string code = recording.GetCode();
                recordingsByCode[code] = recording;
            }

            foreach (var line in File.ReadLines(path))
            {
                var data = line.Split(',');

                Recording recordingCode = recordingsByCode[data[0]];
                double price = double.Parse(data[1]);
                int amount = int.Parse(data[2]);
                DateTime time = DateTime.Parse(data[3]);

                Purchase purchase = new Purchase(recordingCode, price, amount, time);
                purchases.Add(purchase);
            }

            return purchases;
        }


        static List<string> GetAllTitles(string path)
        {
            List<Recording> RecordingsList = LoadRecordings(path);

            List<Piece> PieceList = new List<Piece>();

            foreach (var piece in RecordingsList)
            {
                PieceList.Add(piece.GetPiece());
            }

            List<string> TitleList = new List<string>();
            foreach (var piece in PieceList)
            {
                TitleList.Add(piece.Get_title());
            }
            TitleList = TitleList.Distinct().ToList();

            //foreach (var title in TitleList)
            //{
            //    Console.WriteLine(title);
            //}

            return TitleList;
        }



         //031322 <recording>, 29.69 <price>, 1 <amount>, 2023-01-02 13:47:22<time>
        static string FindMostPopularPiece(string path)
        {
            List<Purchase> purchases = LoadPurchases(path);

            Dictionary<Piece, int> RecordingsPiece = new Dictionary<Piece, int>();

            for (int i = 0; i < purchases.Count; i++)
            {
                var recording = purchases[i].GetRecording().GetPiece();
                if (RecordingsPiece.ContainsKey(recording))
                {
                    RecordingsPiece[recording] += purchases[i].GetAmount();
                }
                else
                {
                    RecordingsPiece[recording] = purchases[i].GetAmount();
                }
            }

            int MostAmount = -1;
            string MostPopularPiece = "";

            foreach (var entry in RecordingsPiece)
            {
                if (entry.Value > MostAmount)
                {
                    MostAmount = entry.Value;
                    MostPopularPiece = entry.Key.Get_title();
                }
            }

            foreach (var entry in RecordingsPiece)
            {
                Console.WriteLine($"Recording ID: {entry.Key.Get_title()}, Total Amount: {entry.Value}");
            }

            return MostPopularPiece;
        }


        static string FindMostPopularComposer(string path)
        {
            List<Purchase> purchases = LoadPurchases(path);

            Dictionary<Composer, int> RecordingsComposer = new Dictionary<Composer, int>();

            for (int i = 0; i < purchases.Count; i++)
            {
                var Composer = purchases[i].GetRecording().GetPiece().GetComposer();
                if (RecordingsComposer.ContainsKey(Composer))
                {
                    RecordingsComposer[Composer] += purchases[i].GetAmount();
                }
                else
                {
                    RecordingsComposer[Composer] = purchases[i].GetAmount();
                }
            }

            int MostAmount = -1;
            string MostPopularComposer = "";

            foreach (var entry in RecordingsComposer)
            {
                if (entry.Value > MostAmount)
                {
                    MostAmount = entry.Value;
                    MostPopularComposer = entry.Key.GetName();
                }
            }

            foreach (var entry in RecordingsComposer)
            {
                Console.WriteLine($"Recording ID: {entry.Key.GetName()}, Total Amount: {entry.Value}");
            }
            return MostPopularComposer;
        }

        static string GetBestSellDay(string path)
        {
            List<Purchase> purchases = LoadPurchases(path);

            Dictionary<DayOfWeek, double> SellsOfTheDay = new Dictionary<DayOfWeek, double>();

            foreach (var purchase in purchases)
            {
                if (SellsOfTheDay.ContainsKey(purchase.GetPurchaseDay().DayOfWeek))
                {
                    SellsOfTheDay[purchase.GetPurchaseDay().DayOfWeek] += purchase.GetTotal();
                }
                else
                {
                    SellsOfTheDay[purchase.GetPurchaseDay().DayOfWeek] = purchase.GetTotal();
                }
            }

            double BestPrice = -1;
            DayOfWeek BestDay = DayOfWeek.Monday;

            foreach (var entry in SellsOfTheDay)
            {
                if(entry.Value > BestPrice)
                {
                    BestPrice = entry.Value;
                    BestDay = entry.Key;
                }
            }

            foreach (var entry in SellsOfTheDay)
            {
                Console.WriteLine($"Recording ID: {entry.Key}, Total Amount: {entry.Value}");
            }

            return BestDay.ToString();
        }

        static double GetAveragePiecePrice(string path, string title)
        {
            List<Purchase> purchases = LoadPurchases(path);

            int TotalAmount = 0;
            double TotalPrice = 0;

            foreach(var item in purchases)
            {
                if(item.GetRecording().GetTitle() == title)
                {
                    TotalAmount += item.GetAmount();
                    TotalPrice += item.GetTotal();
                }
            }

            return TotalPrice/TotalAmount;
        }

        static void Main(string[] args)
        {

            //Composer test = new Composer("Qwerty", "qwerty");
            //Console.WriteLine(test.GetName());
            //test.SetName("1", "Trewq");
            //Console.WriteLine(test.GetName());


            //string filePathPurchases = @"C:\Users\denys\Desktop\My-Pogramming-\TUKE\C#\Assignment1\Assignment1\samples\purchases27.csv";
            //string filePathRecordings = @"C:\Users\denys\Desktop\My-Pogramming-\TUKE\C#\Assignment1\Assignment1\samples\recordings27.csv";
            // write your code here for any tests you want to carry out

            //var myelms = LoadRecordings(filePathRecordings);

            //Console.WriteLine(myelms);

            //var toPrint = GetAllTitles(filePathRecordings);
            //var toPrint = FindMostPopularPiece(filePathPurchases);
            //var toPrint = FindMostPopularComposer(filePathPurchases);
            //var toPrint = GetBestSellDay(filePathPurchases);
            //var toPrint = GetAveragePiecePrice(filePathPurchases, "Sinfonia Concertante for Violin, Viola and Orchestra");
            //Console.WriteLine(toPrint);



            //var toPrint = LoadRecordings(filePathRecordings);
            //Console.WriteLine(toPrint);
            // to test the class structures, use the following instructions:
            //StructureTest sTest = StructureTest.GetInstance();
            //sTest.CheckClasses();
        }
    }
}