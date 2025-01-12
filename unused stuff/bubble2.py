from random import randrange
def bubble_sort(nums):  
    swapped = True
    while swapped:
        swapped = False
        for i in range(len(nums) - 1):
            if nums[i] > nums[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
                swapped = True


list_of_nums = []
n=int(input("enter number of elements: "))
for i in range(n):
    new_val = input(f"enter element {i}: ")
    if new_val == "random" or new_val == "Random":
        list_of_nums = []
        list_of_nums = [randrange(1, 99) for j in range(n)]
        break
    list_of_nums.append(int(new_val))

print("before sorting")
print(list_of_nums)

bubble_sort(list_of_nums)

print("after sorting")
print(list_of_nums)
