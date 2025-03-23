import sys 
one_to_nine = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ten_to_nineteen = ["ten" ,"eleven", "twelve", "thirteen", "fourteen", "fiveteen", "sixteen", "seventeen", "eighteen", "nineteen"]
number = int(input("enter number from 1 to 1000: "))

if number < 0:
    print(f"program doesn\'t work with negative numbers, you have entered {number}")
elif number > 1000:
    print(f"program doesn\'t work with numbers more than 1000, you have entered {number}")

if number < 10:
    print(f"you have entered {number}, it is converted to letters:", one_to_nine[number])
elif number > 9 and number < 20:
    i = number%10
    print (f"you have entered {number}, it is converted to letters:", ten_to_nineteen[i])
elif number > 19 and number < 100:
    i = number // 10
    if i == 2:
        f = number%10
        print(f"you have entered {number}, it is converted to letters:", "twenty", one_to_nine[f])
        sys.exit()
    if i == 3:
        f = number%10
        print(f"you have entered {number}, it is converted to letters:", "thirty", one_to_nine[f])
        sys.exit()
    f=number%10
    print(f"you have entered {number}, it is converted to letters:", one_to_nine[i]+"ty", one_to_nine[f])
elif number > 99 and number < 1000:
    k = number // 100
    i = (number // 10)%10
    x = (number // 10)%10
    if i==1:
        f = number%10
        print(f"you have entered {number}, it is converted to letters:", one_to_nine[k]+"hundred",ten_to_nineteen[f])
        sys.exit()
    if i == 2:
        f = number%10
        print(f"you have entered {number}, it is converted to letters:", one_to_nine[k]+"hundred", "twenty", one_to_nine[f])
        sys.exit()
    if i == 3:
        f = number%10
        print(f"you have entered {number}, it is converted to letters:", one_to_nine[k]+"hundred", "thirty", one_to_nine[f])
        sys.exit()
    f=number%10
    print(f"you have entered {number}, it is converted to letters:", one_to_nine[k]+"hundred", one_to_nine[x]+"ty", one_to_nine[f])
