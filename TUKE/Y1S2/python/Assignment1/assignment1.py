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


def parse_throws(throws):
    separated_list = []
    p1_list, p2_list = [], []
    p1_points, p2_points = 501, 501
    p1_counter_throws = 0  # first player
    p2_counter_throws = 0  # second player

    for throw in throws:
        points = throw_to_points(throw)
        if p1_counter_throws < 3:
            if p1_points - points > 0:
                if p1_points - points == 1:
                    p1_list.append(throw)
                    p1_counter_throws = 3
                else:
                    p1_points -= points
                    p1_list.append(throw)
                    p1_counter_throws += 1
            elif p1_points - points <= 0:
                if p1_points - points < 0:
                    p1_list.append(throw)
                    p1_counter_throws = 3
                else: 
                    p1_list.append(throw)
                    p1_counter_throws = 3   # Player 1 wins this set
                    p2_counter_throws = 3
                    p2_points = 501
                    p1_points = 501
        elif p2_counter_throws < 3:
            if p2_points - points > 0:
                if p2_points - points == 1:
                    p2_list.append(throw)
                    p2_counter_throws = 3
                else: 
                    p2_points -= points
                    p2_list.append(throw)
                    p2_counter_throws += 1
            elif p2_points - points <= 0:
                if p2_points - points < 0:
                    p2_list.append(throw)
                    p2_counter_throws = 3
                else:
                    p2_list.append(throw)
                    p2_counter_throws = 3  # Player 2 wins this set
                    p1_counter_throws = 3
                    p2_points = 501
                    p1_points = 501
        if p1_counter_throws == 3 and p2_counter_throws == 3:
            p1_counter_throws = 0
            p2_counter_throws = 0
            if p1_list and p2_list:
                separated_list.append((tuple(p1_list), tuple(p2_list)))
            elif p1_list:
                separated_list.append((tuple(p1_list),))
            elif p2_list:
                separated_list.append((tuple(p2_list),))
            p1_list.clear()
            p2_list.clear()

    return separated_list




def save_throws(throw_pairs, path):
    file = open(path, 'w')
    list_of_throws = []
 
    for pairs in throw_pairs:
        for item in pairs:
            counter = 0
            for throw in item:
                if throw is None:
                    list_of_throws.append("X")
                else: 
                    list_of_throws.append(throw)
                counter+=1
            if counter != 3:
                while counter != 3:
                    list_of_throws.append("N")
                    counter+=1
        if list_of_throws != 6:
            k = len(list_of_throws)
            while k != 6:
                list_of_throws.append("N")
                k += 1
        # print(list_of_throws)
        for i in range(0,3):
            if list_of_throws[i] == "N" and list_of_throws[3 + i] != "N":
                file.write(f"  {list_of_throws[3+i]}\n")
            elif list_of_throws[3 + i] == "N" and list_of_throws[i] != "N":
                if list_of_throws[3] == "N"  and list_of_throws[4] == "N"  and list_of_throws[5] == "N" :
                    file.write(f"{list_of_throws[i]}\n")
                else: file.write(f"{list_of_throws[i]} \n")
            elif list_of_throws[3 + i] != "N" and list_of_throws[i] != "N":
                file.write(f"{list_of_throws[i]} {list_of_throws[3+i]}\n")
            elif list_of_throws[3 + i] != "N" and list_of_throws[i] != "N":
                file.write(f"{list_of_throws[i]} {list_of_throws[3+i]}\n")
        list_of_throws.clear()
    file.close()



def split_into_sets(throw_pairs):  # 2
    match, sets, legs, turns, turn = [], [], [], [], []
    p1_points, p2_points = 501, 501
    p1_wins, p2_wins = 0, 0
    turns_indicator = 0
    temp = 1
    for throws in throw_pairs:
        for triplet in throws:
            turns_indicator = (turns_indicator +1)%2
            for throw in triplet:
                if turns_indicator == 0:
                    if p1_points - throw_to_points(throw)>=0:
                        p1_points = p1_points - throw_to_points(throw)
                        if p1_points == 1:
                            p1_points +=throw_to_points(throw)
                    turn.append(throw)
                elif turns_indicator == 1:
                    if p2_points - throw_to_points(throw)>=0:
                        p2_points = p2_points - throw_to_points(throw)
                        if p2_points == 1:
                            p2_points +=throw_to_points(throw)
                    turn.append(throw)
            turns.append(tuple(turn.copy()))
            turn = list()
        buffer = tuple(turns.copy())
        match.append(buffer)
        turns = list()
        if p1_wins == 3:
            buffer = list(legs.copy())
            sets.append(buffer)
            p1_wins = p2_wins = 0
            legs = list()
        if p2_wins == 3:
            buffer = list(legs.copy())
            sets.append(buffer)
            p1_wins = p2_wins = 0
            legs = list()
        if p1_points == 0:
            buffer = list(match.copy())
            legs.append(buffer)
            match = list()
            p1_points = p2_points = 501
            p1_wins = p1_wins + 1
            if temp <= 0:
                turns_indicator = 0
                temp = 1
            elif temp > 0:
                turns_indicator = 1
                temp -= temp
        if p2_points == 0:
            buffer = list(match.copy())
            legs.append(buffer)
            match = list()
            p1_points = p2_points = 501
            p2_wins = p2_wins + 1
            if temp == 0:
                turns_indicator = 0
                temp = 1
            elif temp == 1:
                turns_indicator = 1
                temp -= temp
    if len(legs) > 0:
        buffer = list(legs.copy())
        sets.append(buffer)
    return sets



def get_match_result(structured_throws):
    p1_points, p2_points = 501, 501
    P1_wins, P2_wins = 0, 0
    sets_indicator = 1
    p1_won_games, p2_won_games = 0, 0
    leg_wins = []
    
    for games in structured_throws:
        p1_sets_wins, p2_sets_wins = 0, 0
        sets_indicator =(sets_indicator + 1)%2
        leg_indicator = sets_indicator
        for sets in games:
            leg_indicator =(leg_indicator + 1)%2
            player_turns_indicator = leg_indicator
            for lags in sets:
                if P1_wins == 3:
                    P1_wins, P2_wins = 0, 0
                    p1_won_games += 1
                if P2_wins == 3:
                    P1_wins, P2_wins = 0, 0
                    p2_won_games += 1
                for throws in lags:
                    player_turns_indicator = (player_turns_indicator+1)%2
                    for throw in throws:
                        if player_turns_indicator == 0:
                            if p1_points - throw_to_points(throw)>=0:
                                p1_points = p1_points - throw_to_points(throw)
                                if p1_points == 1:
                                    p1_points +=throw_to_points(throw)
                        elif player_turns_indicator == 1:
                            if p2_points - throw_to_points(throw)>=0:
                                p2_points = p2_points - throw_to_points(throw)
                                if p2_points == 1:
                                    p2_points +=throw_to_points(throw)
                if p1_points == 0:
                    p1_points, p2_points = 501, 501
                    p1_sets_wins += 1
                    P1_wins += 1
                elif p2_points == 0:
                    p1_points, p2_points = 501, 501
                    p2_sets_wins += 1
                    P2_wins += 1
                if P1_wins == 3 or P2_wins == 3:
                    leg_wins.append((p1_sets_wins, p2_sets_wins))
                if P1_wins == 3:
                    P1_wins, P2_wins = 0, 0
                    p1_won_games += 1
                if P2_wins == 3:
                    P1_wins, P2_wins = 0, 0
                    p2_won_games += 1
    return ((p1_won_games, p2_won_games), leg_wins)


def get_180s(structured_throws):
    throw_counter_p1, throw_counter_p2 = 0, 0
    amount_180_p1, amount_180_p2 = 0, 0
    count_to_180_p1, count_to_180_p2 = 0, 0
    sets_indicator = 1
    
    for games in structured_throws:
        sets_indicator = (sets_indicator +1)%2
        leg_indicator = sets_indicator
        for sets in games:
            leg_indicator = (leg_indicator+1)%2
            players_turns = leg_indicator
            for legs in sets:
                for pairs in legs:
                    players_turns = (players_turns+1)%2
                    for throws in pairs:
                        if len(pairs)<3:
                            break
                        if players_turns == 0: #player 1
                            count_to_180_p1 = count_to_180_p1 + throw_to_points(throws)
                            throw_counter_p1 += 1
                            if count_to_180_p1 == 180:
                                amount_180_p1 = amount_180_p1 + 1
                                count_to_180_p1, throw_counter_p1 = 0, 0
                                count_to_180_p2, throw_counter_p2 = 0, 0
                            elif throw_counter_p1 == 3:
                                count_to_180_p1, throw_counter_p1 = 0, 0
                                count_to_180_p2, throw_counter_p2 = 0, 0
                                
                        if players_turns == 1: #player 2
                            count_to_180_p2 = count_to_180_p2 + throw_to_points(throws)
                            throw_counter_p2 += 1
                            if count_to_180_p2 == 180:
                                amount_180_p2 = amount_180_p2 + 1
                                count_to_180_p2,  throw_counter_p2 = 0, 0
                                count_to_180_p1, throw_counter_p1 = 0, 0
                            elif throw_counter_p2 == 3:
                                count_to_180_p2, throw_counter_p2 = 0, 0
                                count_to_180_p1, throw_counter_p1 = 0, 0
    return (amount_180_p1, amount_180_p2)


def get_average_throws(structured_throws):  # 1
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
                            if p1_points - throw_to_points(throws)>=0:
                                p1_points = p1_points - throw_to_points(throws)
                                if p1_points == 1:
                                    p1_points +=throw_to_points(throws)
                        if players_turns == 1:
                            if p2_points - throw_to_points(throws)>=0:
                                p2_points = p2_points - throw_to_points(throws)
                                if p2_points == 1:
                                    p2_points +=throw_to_points(throws)
                rest_p1_points = 501 - p1_points
                rest_p2_points = 501 - p2_points
                # print(rest_p1_points, rest_p2_points)
                if p1_points == 0:
                    p1_points, p2_points = 501, 501
                    p1_wins+=1
                if p2_points == 0:
                    p1_points, p2_points = 501, 501
                    p2_wins+=1
                if p1_wins == 3 or p2_wins == 3:
                    p1_wins, p2_wins = 0, 0
            total_sum_p1 +=rest_p1_points
            total_sum_p2 +=rest_p2_points

        avarages_p1=total_sum_p1/count_pairs_p1
        avarages_p2=total_sum_p2/count_pairs_p2
        tup=(avarages_p1,avarages_p2)
        lst_of_avarages.append(tup)
        total_sum_p1, total_sum_p2 = 0, 0
        count_pairs_p1, count_pairs_p2 = 0, 0
    return lst_of_avarages


def generate_finishes(points):  # 2
   
    result = []
    scores = {}
    scores_double = {}
    scores = {str(i): i for i in range(points) if i <= 21}
    scores.update({f'D{i}': i * 2 for i in range(1, 21) if i * 2 <= points})
    scores.update({f'T{i}': i * 3 for i in range(1, 21) if i * 3 <= points})
    scores_double = {f'D{i}': int(i) * 2 for i in range(1, 21) if int(i) * 2 <= points}
    scores_double['50'] = 50
    scores['25'] = 25
    scores['50'] = 50
        
    for value1_to_add,value1_to_check in scores.items():
        for value2_to_add,value2_to_check in scores.items():
            for key,value in scores_double.items():
                if value1_to_check + value2_to_check + value == points:
                    if [value2_to_add,value1_to_add,key] in result:
                        continue
                    elif [value1_to_add,key] in result:
                        continue
                    elif [value2_to_add,key] in result:
                        continue
                    elif value1_to_check == 0 and value2_to_check == 0:
                        result.append([key])
                    elif value1_to_check == 0:
                        result.append([value2_to_add,key])
                    elif value2_to_check == 0:
                        result.append([value1_to_add,key])
                    else:
                        result.append([value1_to_add,value2_to_add,key])
    return result

if __name__ == '__main__':
    throw=[(('T20', 'T20', 'T20'), ('T20', 'T20', '5')), (('T20', 'T20', 'T20'), ('1', 'T20', 'T20')), (('T20', '19', 'T20'), ('1', '20', '20')), (('1', 'D1'),)]
    # throw2=[['T20', 'T20', 'D4'], ['T18', 'T20', 'D7'], ['T19', 'T19', 'D7'], ['T20', '50', 'D9'], ['T16', 'T20', 'D10'], ['T17', 'T19', 'D10'], ['T18', 'T18', 'D10'], ['T18', '50', 'D12'], ['T14', 'T20', 'D13'], ['T15', 'T19', 'D13'], ['T16', 'T18', 'D13'], ['T17', 'T17', 'D13'], ['D20', 'T20', 'D14'], ['50', '50', 'D14'], ['D19', 'T20', 'D15'], ['T16', '50', 'D15'], ['D18', 'T20', 'D16'], ['T12', 'T20', 'D16'], ['T13', 'T19', 'D16'], ['T14', 'T18', 'D16'], ['T15', 'T17', 'D16'], ['T16', 'T16', 'D16'], ['D17', 'T20', 'D17'], ['D20', 'T18', 'D17'], ['D16', 'T20', 'D18'], ['D19', 'T18', 'D18'], ['T14', '50', 'D18'], ['D15', 'T20', 'D19'], ['D18', 'T18', 'D19'], ['D20', '50', 'D19'], ['T10', 'T20', 'D19'], ['T11', 'T19', 'D19'], ['T12', 'T18', 'D19'], ['T13', 'T17', 'D19'], ['T14', 'T16', 'D19'], ['T15', 'T15', 'D19'], ['D14', 'T20', 'D20'], ['D17', 'T18', 'D20'], ['D19', '50', 'D20'], ['D20', 'T16', 'D20'], ['18', 'T20', '50'], ['D9', 'T20', '50'], ['D12', 'T18', '50'], ['D14', '50', '50'], ['D15', 'T16', '50'], ['D18', 'T14', '50'], ['D19', 'D20', '50'], ['T6', 'T20', '50'], ['T7', 'T19', '50'], ['T8', 'T18', '50'], ['T9', 'T17', '50'], ['T10', 'T16', '50'], ['T11', 'T15', '50'], ['T12', 'T14', '50'], ['T13', 'T13', '50']]
    print(split_into_sets(throw))
    pass
