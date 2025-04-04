﻿using System;

namespace Composite
{
    //Leaf
    public class Commit : GitComponent
    {
        private readonly string _commitId;

        public Commit(string commitId)
        {
            _commitId = commitId;
        }
        public override void ShowDetail()
        {
            Console.WriteLine($"- Commit ID: {this._commitId}");
        }
    }
}
