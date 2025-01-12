using System;

namespace Composite.Structural
{

    public class Program
    {
        //https://www.youtube.com/watch?v=-4ex4VKvQus&pp=ygUbQyMgQ29tcG9zaXRlIERlc2lnbiBQYXR0ZXJu
        //client
        public static void Main(string[] args)
        {
            GitComponent mainBranch = new Branch("main");
            GitComponent commitToMain1 = new Commit("246hbb943mb");
            GitComponent commitToMain2 = new Commit("056nn8fs;h8");
            GitComponent commitToMain3 = new Commit("nldis965m43");
            GitComponent commitToMain4 = new Commit("msnpeloof94");
            mainBranch.Add(commitToMain1);
            mainBranch.Add(commitToMain2);
            mainBranch.Add(commitToMain3);
            mainBranch.Add(commitToMain4);
            mainBranch.Remove(commitToMain4);

            GitComponent smallFeature = new Branch("small-feature");
            mainBranch.Add(smallFeature);

            GitComponent smallFeature1 = new Commit("oa092n23lb8");
            smallFeature.Add(smallFeature1);

            mainBranch.ShowDetail();
            Console.WriteLine("------------------------------");
            commitToMain1.ShowDetail();

            Console.ReadKey();
        }
    }
}
