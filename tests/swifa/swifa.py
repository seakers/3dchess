import os
import pandas as pd

def get_place_points(place : int) -> int:
    points_per_place = [
                        1000,   #1
                        920,    #2
                        850,    #3
                        840,    #4
                        700,    #5
                        695,    #6
                        690,    #7
                        685,    #8
                        680,    #9
                        675,    #10
                        670,    #11
                        665,    #12
                        660,    #13
                        655,    #14
                        650,    #15
                        645 ,   #16
                        505 ,   #17
                        500 ,   #18
                        495 ,   #19
                        490 ,   #20
                    ]
    
    return points_per_place[place-1] if place <= len(points_per_place) else 0

def load_results(tournament_number : int, results_path : str, trounament_name : str = 'swifa') -> str:
    """
    Loads the results of a SWIFA tournament from the specified path.
    
    Args:
        tournament_number (int): The tournament number to load.
        results_path (str): The path to the results directory.
        trounament_name (str): The name of the tournament. Defaults to 'swifa'.
    
    Returns:
        str: The path to the loaded results.
    """
    file_name = f"{trounament_name}_{tournament_number}.csv"
    tournament_path = os.path.join(results_path, file_name)
    
    if not os.path.exists(tournament_path):
        raise FileNotFoundError(f"Tournament {tournament_number} does not exist in {results_path}.")
    
    tournament_data = pd.read_csv(tournament_path,delimiter=', ',engine='python')
    tournament_data['Tournament Number'] = tournament_number

    return tournament_data

def calculate_scores(tournament_data : pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the scores for each player in the tournament based on their placement.
    
    Args:
        tournament_data (pd.DataFrame): The DataFrame containing tournament results.
    
    Returns:
        pd.DataFrame: The DataFrame with calculated scores.
    """
    
    tournaments = tournament_data['Tournament Number'].unique()
    universities = tournament_data['University'].unique()
    scores = {university: 0 for university in universities}
    weapons = tournament_data['Weapon'].unique()

    for university in universities:
        if 'dallas' in university.lower(): 
            x = 1

        for weapon in weapons:
            for tournament in tournaments:
                placements : pd.DataFrame = tournament_data[(tournament_data['Tournament Number'] == tournament) & 
                                                    (tournament_data['Weapon'] == weapon) & 
                                                    (tournament_data['University'] == university)]
                placements = placements.sort_values(by='Place')

                if placements.empty: continue # No placements for this university in this tournament

                squads = placements['Squad'].unique()
                assert len(placements) == len(squads), f"Number of entries ({len(placements)}) does not match number of squads ({len(squads)}) for {university} in SWIFA {tournament} with weapon {weapon}."

                best_placement = int(placements['Place'].values[0])
                score = get_place_points(best_placement)
                scores[university] += score
    
    leaderboard = [(university,scores[university]) for university in scores]
    leaderboard = sorted(leaderboard, key=lambda x: x[1], reverse=True)
    return pd.DataFrame(leaderboard, columns=['University', 'Score'])


def main(desired_winner : str = 'Texas A&M University', results_path : str = './results'):
    df = None
    cumulative_scores = None

    for i in range(1, 4):
        tournament_data = load_results(i, results_path)
        
        if df is None:
            df = tournament_data
        else:
            df = pd.concat([df, tournament_data], ignore_index=True)

        tournament_scores = calculate_scores(tournament_data)
        cumulative_scores = calculate_scores(df)

        print(f'\n============= SWIFA {i} =============')
        print('\nTournament Scores:')
        print(tournament_scores)
        print('\nCumulative Scores:')
        print(cumulative_scores)
                           
    print(f'\n=====================================')
    lead = cumulative_scores.values[0][1] - cumulative_scores.values[1][1]
    print(f'Leader: {cumulative_scores.values[0][0]} (+{lead})')

    # GOAL see what placements give a university the win
    
    x = 1
                    

if __name__ == "__main__":
    main()