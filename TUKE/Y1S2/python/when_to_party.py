test_intervals1 = [
    ('Daniel', 7, 9),
    ('Palo', 8, 11),
    ('Danka', 10, 12),
    ('Miro', 9, 1),
    ('Lubos', 11, 12),
    ('Milan', 10, 12),
    ('Noro', 9, 11),
    ('Radka', 8, 10),
    ('Martina', 7, 8),
    ('Peto', 9, 11)
]
test_intervals2 = [
    ('Daniel', 10, 12),
    ('Palo', 9, 12),
    ('Danka', 10, 12),
    ('Miro', 9, 1),
    ('Lubos', 10, 11),
    ('Milan', 11, 2),
    ('Noro', 10, 1),
    ('Radka', 9, 12),
    ('Martina', 9, 11),
    ('Peto', 9, 1),
    ('Julka', 10, 12),
    ('Riso', 10, 11),
    ('Mato', 11, 3),
    ('Brano', 10, 1),
    ('David', 9, 11)
]
test_intervals_weights = [
    ('Daniel', 8, 10, 1),
    ('Palo', 8, 11, 2),
    ('Danka', 8, 9, 3),
    ('Miro', 11, 1, 4),
    ('Lubos', 9, 12, 2),
    ('Milan', 10, 12, 2),
    ('Noro', 10, 3, 1),
    ('Radka', 11, 12, 2),
    ('Martina', 10, 12, 1),
    ('Peto', 8, 11, 2)
]


def load_intervals(intervals):
    lst = []
    # returns a list of intervals as tuples
    for _, start, end in intervals:
        if start > end:
            end += 12
        lst.append((start, end))
    return lst

def check_intervals(intervals):
    for elem in intervals:
        if not isinstance(intervals, list):
            raise TypeError(f"expected list, got{type(intervals)}")
        if len(elem) != 4 and len(elem) !=3:
            raise ValueError(f"Tuples must have 3 elements")
        name, start, end = elem
        if not isinstance(name, str):
            raise TypeError(f"expected string, got{type(name)}")
        if not isinstance(start, int):
            raise TypeError(f"expected int, got{type(start)}")
        if not isinstance(end, int):
            raise TypeError(f"expected int, got{type(end)}")
        if len(elem == 4 and not isinstance(elem[3], int)):
            print("WARNING")
def choose_time(interval_list):
    # return best time to party and number of friends present
    events = []
    for start, end in interval_list:
        events.append((start, 'start'))
        events.append((end, 'end'))
    max_count = 0
    rcount=0
    time=0
    for evt_time, evt_type in events:
        if evt_type == 'start':
            rcount+=1
        elif evt_type == 'end':
            rcount-=1
        if rcount>max_count:
            max_count=rcount
            time = evt_time
    return time, max_count


def when_to_go(friends_list, y_start=None, y_end=None):
    # process intervals from list
    # call function to calculate best time for party
    # print results
    if y_start is not None and y_end is not None:
        best_time, max_count = choose_time_constrained(friends_list, y_start, y_end)
    else:
        intervals = load_intervals(friends_list)
    best_time, max_count = choose_time
    print(f"The best time to attend is at {best_time} oclock when youll meet {max_count}")
    pass


def choose_time_constrained(interval_list, y_start, y_end):
    # return best time to party and number of friends present
    # look only in interval <y_start, y_end)
    events = []
    for start, end in interval_list:
        if start in range(y_start, y_end):
            events.append((start, 'start'))
        if end in range(y_start, y_end):
            events.append((end, 'end'))
    max_count = 0
    rcount=0
    time=0
    for evt_time, evt_type in events:
        if evt_type == 'start':
            rcount+=1
        elif evt_type == 'end':
            rcount-=1
        if rcount>max_count:
            max_count=rcount
            time = evt_time
    pass


def choose_time_with_weights(interval_list):
    start_value = 5
    # return best time to party and total weight
    # for _, start, end, value in interval_list:
    pass

print(load_intervals(test_intervals1))