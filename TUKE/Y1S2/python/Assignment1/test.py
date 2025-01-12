def throw_to_points(throw):
    points = 0
    if throw is None:
        return points
    if throw.isdigit():
        points = int(throw)
    elif throw[0] == 'D':
        num = throw[1:]
        points = int(num) * 2
    elif throw[0] == 'T':
        num = throw[1:]
        points = int(num) * 3
    return points

structured_throws = [[[(('T20', 'T20', 'T20'), ('20', 'T1', 'T1')), (('T20', 'T20', 'T5'), ('5', 'T1', '20')), (('T20', '1', 'T20'), ('T20', '5', '1')), (('T19', 'D4'),)], [(('T20', '5', 'T20'), ('T20', 'T20', 'T20')), (('T20', 'T20', '5'), ('T20', 'T20', '20')), (('20', '1', '1'), ('5', 'T20', 'T20')), (('T20', 'T20', '1'), ('T18', 'D1'))], [(('T1', 'T20', 'T1'), ('20', 'T20', 'T20')), (('T20', 'T20', '1'), ('T20', '1', 'T20')), (('1', 'T20', '5'), ('T1', '1', '5')), (('T20', 'T5', '5'), ('1', 'T20', 'T20')), (('1', 'T20', 'T20'), ('T20', '50'))], [(('5', 'T20', '5'), ('20', 'T20', '5')), (('1', 'T20', '1'), ('T20', '20', 'T20')), (('5', 'T20', 'T20'), ('T20', 'T20', 'T20')), (('T20', 'T20', '20'), ('T20', '1', 'T11')), (('T20', '9', '11'), ('D1',))]], [[(('T20', 'T20', '1'), ('T20', '5', 'T20')), (('T20', 'T20', '20'), ('T20', 'T20', 'T1')), (('T20', 'T20', '20'), ('T20', '1', 'T20')), (('T20', 'D20'),)], [(('T20', 'T20', 'T20'), ('T20', 'T20', 'T20')), (('20', 'T20', '20'), ('1', 'T20', '1')), (('T20', 'T20', 'T20'), ('T20', '1', 'T5')), (('T13', 'D1'),)], [(('T20', '1', 'T20'), ('20', 'T20', 'T20')), (('T5', '5', 'T20'), ('5', 'T20', '20')), (('T1', '20', 'T20'), ('T20', '1', '5')), (('T20', 'T20', 'T19'), ('20', 'T20', 'T20')), (('D20',),)], [(('T20', 'T20', 'T20'), ('T20', 'T20', '5')), (('T20', 'T20', 'T20'), ('T20', 'T20', '20')), (('T20', 'T19', 'D12'),)], [(('20', 'T20', '20'), ('T20', 'T20', 'T20')), (('T20', 'T1', 'T20'), ('T20', 'T20', 'T20')), (('5', '20', 'T1'), ('T20', 'T19', 'D12'))]], [[(('1', 'T20', 'T20'), ('T20', 'T20', 'T20')), (('T20', 'T20', '1'), ('T20', 'T20', 'T20')), (('T20', 'T20', 'T1'), ('T20', 'T19', '9')), (('T5', 'T20', 'T19'), ('13', 'D1'))], [(('T20', 'T20', 'T20'), ('T20', '20', 'T20')), (('5', 'T20', '20'), ('T20', 'T20', 'T20')), (('T20', 'T20', '5'), ('T20', 'T20', 'T19')), (('T20', '10', 'T13'), ('D2',))], [(('T20', 'T20', 'T20'), ('T20', 'T20', 'T20')), (('T20', 'T20', 'T20'), ('5', 'T20', 'T20')), (('T20', 'T19', 'D12'),)], [(('T20', 'T20', '20'), ('T20', 'T20', '20')), (('20', 'T20', 'T20'), ('T20', 'T20', 'T20')), (('1', 'T5', 'T20'), ('T20', '1', 'T1')), (('T20', 'T19', 'D14'),)], [(('5', '5', 'T20'), ('T20', '1', '20')), (('T20', 'T20', 'T20'), ('T20', 'T20', '5')), (('20', 'T20', 'T20'), ('20', 'T20', '1')), (('T20', 'T15', 'D3'),)]]]
# p1_points, p2_points = 501, 501
# p1_wins, p2_wins = 0, 0
# counter = 0
# counter_sets = 1
# total_sum_p1 = 0
# tutal_sum_p2 = 0
# w = 0
# q = 0
# co = 0
# average = []

# for games in structured_throws:
#     set_f = 0
#     set_s = 0
#     counter_sets=(counter_sets+1)%2
#     counter = counter_sets
#     for sets in games:  
#         counter=(counter+1)%2
#         p1 = counter
#         for legs in sets: 
#             if p1_wins == 3:
#                 p1_wins = 0
#                 p2_wins = 0
#                 total_sum_p1+=1 
#             if p2_wins == 3:
#                 p1_wins = 0
#                 p2_wins = 0
#                 tutal_sum_p2+=1
#             for items in legs:
#                 p1 = (p1+1)%2 
#                 if p1 == 0:
#                     w+=1
#                 else: q+=1    
#                 for item in items:
#                     if p1 == 0:
#                         p1_points -= throw_to_points(item)
#                         if p1_points == 1:
#                             p1_points+=throw_to_points(item)
#                         if p1_points < 0:
#                             p1_points+=throw_to_points(item)
#                         print(f"{p1_points} - f")
#                     elif p1 == 1:
#                         p2_points -= throw_to_points(item)
#                         if p2_points == 1:
#                             p2_points+=throw_to_points(item)
#                         if p2_points < 0:
#                             p2_points+=throw_to_points(item)
#                         print(f"{p2_points} - s")
#             set_f = 501 - p1_points
#             set_s = 501 - p2_points
#             print("set", set_f)
#             if p1_points == 0:
#                 print("f")
#                 p1_points = 501
#                 p2_points = 501
#                 p1_wins+=1
#             if p2_points == 0:
#                 p2_points = 501
#                 p1_points = 501
#                 p2_wins+=1
#             if p1_wins == 3:
#                 p1_wins = 0
#                 p2_wins = 0
#             if p2_wins == 3:
#                 p1_wins = 0
#                 p2_wins = 0
#         print(w)
#         total_sum_p1 += set_f 
#         tutal_sum_p2 += set_s 
#         print(total_sum_p1)
       
       
#     average.append((total_sum_p1 / w, tutal_sum_p2 / q))
#     total_sum_p1 = 0
#     tutal_sum_p2 = 0
#     w = 0
#     q = 0
# print(average)

p1_points, p2_points = 501, 501
total_sum_p1, total_sum_p2 = 0, 0
count_pairs_p1, count_pairs_p2 = 0, 0
p1_wins, p2_wins = 0, 0
sets_indicator = 1
lst_of_avarages = []

for games in structured_throws:
    rest_p1_points, rest_p2_points = 0, 0
    sets_indicator = (sets_indicator +1)%2
    leg_indicator = sets_indicator
    for sets in games:
        leg_indicator = (leg_indicator+1)%2
        players_turns = leg_indicator
        for legs in sets:
            for pairs in legs:
                players_turns = (players_turns+1)%2
                if players_turns == 0:
                    count_pairs_p1+=1
                else: count_pairs_p2+=1
                for throws in pairs:
                    if players_turns == 0:
                        p1_points -= throw_to_points(throws)
                        if p1_points == 1:
                            p1_points += throw_to_points(throws)
                        if p1_points < 0:
                            p1_points +=throw_to_points(throws)
                    if players_turns == 1:
                        p2_points -= throw_to_points(throws)
                        if p2_points == 1:
                            p2_points+=throw_to_points(throws)
                        if p2_points < 0:
                            p2_points+=throw_to_points(throws)
            rest_p1_points = 501 - p1_points
            rest_p2_points = 501 - p2_points
            if p1_points == 0:
                p1_points = 501
                p2_points = 501
                p1_wins+=1
            if p2_points == 0:
                p2_points = 501
                p1_points = 501
                p2_wins+=1
            if p1_wins == 3 or p2_wins == 3:
                p1_wins = 0
                p2_wins = 0
        total_sum_p1 +=rest_p1_points
        total_sum_p2 +=rest_p2_points
        
    avarages_p1=total_sum_p1/count_pairs_p1
    avarages_p2=total_sum_p2/count_pairs_p2
    tup=(avarages_p1,avarages_p2)
    lst_of_avarages.append(tup)
    total_sum_p2 = 0
    total_sum_p1 = 0
    count_pairs_p2 = 0
    count_pairs_p1 = 0
print(lst_of_avarages)