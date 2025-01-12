def load_results(file_path):
    # open file and read results from it
    # returns a list of rows as strings
    with open(file_path, 'r') as f:
        return f.readlines()


def load_score(line):
    # loads score from a row as string
    # returns a tuple of four values: home team name, home team goals
    # away team name, away team goals
    home_info, away_info = line.strip().split(' - ')
    home_team, home_goals = home_info.rsplit(' ', maxsplit=1)
    away_goals, away_team = away_info.split(' ', maxsplit=1)
    
    return (home_team, int(home_goals), away_team, int(away_goals))


def get_team_list(scores):
    # returns a set of unique team names
    return set([score[0] for score in scores])



def create_table(team_names):
    # creates an empty table with team names
    # there is one dictionary for every team with the following info:
    # team name, number of games played, number of wins, number of draws
    # number of losses, number of goals scored, number of goals conceeded
    # number of points
    row = {
        "name": "",
        "played": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "goals_for": 0,
        "goals_against": 0,
        "points": 0,
    }
    table = []
    for team in team_names:
        team_row = row.copy()
        team_row['name'] = team
        table.append(team_row)
    return table


def get_team_row(table, team_name):
    # finds the dictionary corresponding to the team in a table
    for row in table:
        if row['name'] == team_name:
            return row
    return None


def update_points(table):
    # calculates the team's points based on number of wins=3, draws=1, losses=0
    for row in table:
        row['points'] = 3 * row['wins'] + row['draws']


def add_match_to_table(table, match_info):
    # updates table based on a match result
    # input is a list representing the table and
    # a tuple representing the match result
    # updates table directly, returns nothing
    h_team, h_goals, a_team, a_goals = match_info
    h_row = get_team_row(table, h_team)
    a_row = get_team_row(table, a_team)
    h_row['played'] += 1
    a_row["played"] += 1
    h_row['wins'] = h_goals > a_goals
    a_row['wins'] = h_goals < a_goals
    h_row['draws'] = h_goals == a_goals
    a_row['draws'] = h_goals == a_goals
    h_row['losses'] = h_goals < a_goals
    a_row['losses'] = h_goals > a_goals
    
    h_row['goals_for'] = h_goals
    h_row['goals_against'] = a_goals
    
    a_row['goals_for'] = a_goals
    a_row['goals_against'] = h_goals


def generate_table(results_file_path):
    # gets path to file with results
    # generates table and fills it with values based on match results
    # returns the table as a list of dictionaries
    pass


def print_table(table):
    # prints table in a user-friendly way
    # columns:
    # rank, team, games, wins, draws, losses, goals for, goals against, points
    # does not return anything
    pass


def sort_table(table):
    # sorts table based on a key, returns a copy of the table
    pass


if __name__ == '__main__':
    path_buli = "C:/Users/denys/OneDrive/Desktop/Programming/TUKE/S2/python/buli_results.txt"
    table = generate_table("buli_results.txt")
    print(load_score('SV Werder Bremen 0 - 4 FC Bayern Mnchen\n'))
    # results = load_results(path_buli)
    # print(results)
    
    table = sort_table(table)

    print_table(table)
