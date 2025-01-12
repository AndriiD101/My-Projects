#------------------------------------------------------------------------------
def deviders1(num: int = 0):
    lst = []
    for i in range(num+1):
        if i%3 == 0 and i%5 == 0:
            lst.append(i)
    return lst
#------------------------------------------------------------------------------
def deviders2(num: int = 0):
    return [i for i in range(num + 1) if i % 3 == 0 and i % 5 == 0]

