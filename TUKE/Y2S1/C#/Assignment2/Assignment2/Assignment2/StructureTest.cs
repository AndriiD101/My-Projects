using Assignment2.Clerks;
using Assignment2.Clients;
using Assignment2.Forms;
using System.Reflection;

namespace Assignment2
{
    public class StructureTest
    {
        private Dictionary<string, FieldInfo> formFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> agendaFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> abstractClerkFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> rudeClerkFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> kindClerkFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> impatientClerkFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> clientFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> eagerClientFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> carefulClientFields = new Dictionary<string, FieldInfo>();
        private Dictionary<string, FieldInfo> optimizingClientFields = new Dictionary<string, FieldInfo>();

        private Dictionary<string, Dictionary<string, FieldInfo>> classOverview = new Dictionary<string, Dictionary<string, FieldInfo>>();

        public bool PublicFound { get; set; } = false;

        private static StructureTest instance = new StructureTest();

        private StructureTest()
        {
            classOverview.Add("Form", formFields);
            classOverview.Add("Agenda", agendaFields);
            classOverview.Add("AbstractClerk", abstractClerkFields);
            classOverview.Add("RudeClerk", rudeClerkFields);
            classOverview.Add("KindClerk", kindClerkFields);
            classOverview.Add("ImpatientClerk", impatientClerkFields);
            classOverview.Add("Client", clientFields);
            classOverview.Add("EagerClient", eagerClientFields);
            classOverview.Add("CarefulClient", carefulClientFields);
            classOverview.Add("OptimizingClient", optimizingClientFields);
        }

        public static StructureTest GetInstance()
        {
            return instance;
        }

        private void CheckForLocal(Type checkType, string typeName)
        {
            FieldInfo[] publicInfos = checkType.GetFields(BindingFlags.Public | BindingFlags.Instance);
            if (publicInfos.Length != 0)
            {
                PublicFound = true;
                Console.WriteLine($"\tPublic field found in {typeName}:");
                foreach (FieldInfo field in publicInfos)
                {
                    Console.WriteLine($"\t\t{field.Name}");
                }
            }
        }

        private bool FindField(FieldInfo[] classFields, string fieldName)
        {
            foreach (FieldInfo field in classFields)
            {
                if (field.Name == fieldName)
                {
                    return true;
                }
            }

            return false;
        }

        private void CheckClassStructure(Type checkType, string typeName, string[] fields)
        {
            CheckForLocal(checkType, typeName);

            FieldInfo[] classFields = checkType.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);

            List<Type> subclasses = new List<Type>() { typeof(RudeClerk), typeof(KindClerk), typeof(ImpatientClerk), typeof(EagerClient), typeof(CarefulClient), typeof(OptimizingClient) };
            if (subclasses.Contains(checkType))
            {
                FieldInfo[] superFields = checkType.BaseType.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
                classFields = classFields.Concat(superFields).ToArray();
            }

            //foreach (FieldInfo info in classFields)
            //{
            //    Console.WriteLine($"{info.Name}");
            //}

            if (fields.Length > classFields.Length)
            {
                Console.WriteLine($"\tMissing fields in {typeName}, expected at least {fields.Length}, got {classFields.Length}");
                return;
            }



            foreach (string correctField in fields)
            {
                if (!FindField(classFields, correctField))
                {
                    Console.WriteLine($"\tMissing field {correctField} in {typeName}");
                    return;
                }

                foreach (FieldInfo field in classFields)
                {
                    if (field.Name == correctField && !classOverview[typeName].ContainsKey(field.Name))
                    {
                        classOverview[typeName].Add(correctField, field);
                    }
                }
            }

            Console.WriteLine($"\t{typeName} structure check finished: OK");
        }

        public void CheckClasses()
        {
            Console.WriteLine("Class structure check starting");

            CheckClassStructure(typeof(Form), "Form", new string[] { "difficulty", "filledOut", "formTitle", "handlingTime" });
            CheckClassStructure(typeof(Agenda), "Agenda", new string[] { "forms", "handled", "type" });

            CheckClassStructure(typeof(AbstractClerk), "AbstractClerk", new string[] { "agendas", "speed", "queue" });
            CheckClassStructure(typeof(RudeClerk), "RudeClerk", new string[] { "agendas", "speed", "queue" });
            CheckClassStructure(typeof(KindClerk), "KindClerk", new string[] { "agendas", "speed", "queue" });
            CheckClassStructure(typeof(ImpatientClerk), "ImpatientClerk", new string[] { "agendas", "speed", "queue", "limit" });

            CheckClassStructure(typeof(Client), "Client", new string[] { "ability", "agenda", "visited" });
            CheckClassStructure(typeof(EagerClient), "EagerClient", new string[] { "ability", "agenda", "visited" });
            CheckClassStructure(typeof(CarefulClient), "CarefulClient", new string[] { "ability", "agenda", "visited" });
            CheckClassStructure(typeof(OptimizingClient), "OptimizingClient", new string[] { "ability", "agenda", "visited" });

            Console.WriteLine("Class structure check finished\n");
        }

        public object GetFieldValue(object obj, string objectClass, string fieldName)
        {
            try
            {
                return classOverview[objectClass][fieldName].GetValue(obj);
            }
            catch (KeyNotFoundException ex)
            {
                return null;
            }
        }

        public void SetFieldValue(object obj, string objectClass, string fieldName, object value)
        {
            classOverview[objectClass][fieldName].SetValue(obj, value);
        }
    }
}
