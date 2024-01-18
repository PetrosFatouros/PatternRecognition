import pandas as pd
import sqlite3

pd.options.mode.chained_assignment = None  # suppress SettingWithCopyWarning


def filter_data():
    # Read sqlite query results into a pandas DataFrame (Match)
    # https://www.kaggle.com/datasets/hugomathien/soccer
    connection = sqlite3.connect('database.sqlite')
    match_df = pd.read_sql_query('SELECT * from Match', connection)

    # Remove rows that contain odds with NaN values
    match_df = match_df.dropna(subset=['B365H', 'B365D', 'B365A',
                                       'BWH', 'BWD', 'BWA',
                                       'IWH', 'IWD', 'IWA',
                                       'LBH', 'LBD', 'LBA'])

    # Export Match dataframe to csv
    match_df.to_csv("Match.csv", index=False)

    # Create a DataFrame with the results of each match
    # 0 -> draw (D)
    # 1 -> home team wins (H)
    # 2 -> away team wins (A)
    match_results_df = match_df[['match_api_id', 'home_team_api_id', 'away_team_api_id']]
    match_results_df['result'] = match_df['home_team_goal'] - match_df['away_team_goal']
    match_results_df.loc[match_results_df['result'] == 0, 'result'] = 0
    match_results_df.loc[match_results_df['result'] > 0, 'result'] = 1
    match_results_df.loc[match_results_df['result'] < 0, 'result'] = 2

    # Export dataframe to csv
    match_results_df.to_csv('match_results.csv', index=False)

    # Create 4 DataFrames with the odds of each betting company
    B365_odds_df = match_df[['match_api_id', 'home_team_api_id', 'away_team_api_id', 'B365H', 'B365D', 'B365A']]
    BW_odds_df = match_df[['match_api_id', 'home_team_api_id', 'away_team_api_id', 'BWH', 'BWD', 'BWA']]
    IW_odds_df = match_df[['match_api_id', 'home_team_api_id', 'away_team_api_id', 'IWH', 'IWD', 'IWA']]
    LB_odds_df = match_df[['match_api_id', 'home_team_api_id', 'away_team_api_id', 'LBH', 'LBD', 'LBA']]

    # Export dataframes to csv
    B365_odds_df.to_csv('B365_odds.csv', index=False)
    BW_odds_df.to_csv('BW_odds.csv', index=False)
    IW_odds_df.to_csv('IW_odds.csv', index=False)
    LB_odds_df.to_csv('LB_odds.csv', index=False)

    # Read sqlite query results into a pandas DataFrame (Team_Attributes)
    connection = sqlite3.connect('database.sqlite')
    team_attributes_df = pd.read_sql_query('SELECT * from Team_Attributes', connection)

    # Remove rows that contain odds with NaN values
    team_attributes_df = team_attributes_df.dropna(subset=['buildUpPlaySpeed', 'buildUpPlayPassing',
                                                           'chanceCreationPassing', 'chanceCreationCrossing',
                                                           'chanceCreationShooting',
                                                           'defencePressure', 'defenceAggression', 'defenceTeamWidth'])

    # Export Team_Attributes dataframe to csv
    team_attributes_df.to_csv("Team_Attributes.csv", index=False)

    # Create a DataFrame with each team's 8 characteristics
    team_characteristics_df = team_attributes_df[['team_api_id', 'buildUpPlaySpeed', 'buildUpPlayPassing',
                                                  'chanceCreationPassing', 'chanceCreationCrossing',
                                                  'chanceCreationShooting',
                                                  'defencePressure', 'defenceAggression', 'defenceTeamWidth']]

    # Multiple records correspond to the same team (based on different dates)
    # Merge these records by calculating the average value of each team's attributes
    column_names = ['buildUpPlaySpeed', 'buildUpPlayPassing',
                    'chanceCreationPassing', 'chanceCreationCrossing', 'chanceCreationShooting',
                    'defencePressure', 'defenceAggression', 'defenceTeamWidth']
    team_characteristics_df = team_characteristics_df.groupby('team_api_id', as_index=False)[column_names].mean()

    # Export dataframe to csv
    team_characteristics_df.to_csv('team_characteristics.csv', index=False)

    # Create a DataFrame which represents the input layer of the multi-layer neural network

    # Define column names
    column_names = ['H_buildUpPlaySpeed', 'H_buildUpPlayPassing',  # home team characteristics
                    'H_chanceCreationPassing', 'H_chanceCreationCrossing', 'H_chanceCreationShooting',
                    'H_defencePressure', 'H_defenceAggression', 'H_defenceTeamWidth',
                    'A_buildUpPlaySpeed', 'A_buildUpPlayPassing',  # away team characteristics
                    'A_chanceCreationPassing', 'A_chanceCreationCrossing', 'A_chanceCreationShooting',
                    'A_defencePressure', 'A_defenceAggression', 'A_defenceTeamWidth',
                    'B365H', 'B365D', 'B365A',  # B365 odds
                    'BWH', 'BWD', 'BWA',  # BWH odds
                    'IWH', 'IWD', 'IWA',  # IW odds
                    'LBH', 'LBD', 'LBA']  # LBH odds

    # Gather the data of the DataFrame
    data = []
    for index, row in match_results_df.iterrows():
        record = []

        # home team attributes
        home_team_id = row['home_team_api_id']
        home_team_attributes_df = team_characteristics_df.loc[team_characteristics_df['team_api_id'] == home_team_id]
        record.append(home_team_attributes_df.values.tolist())

        # away team attributes
        away_team_id = row['away_team_api_id']
        away_team_attributes_df = team_characteristics_df.loc[team_characteristics_df['team_api_id'] == away_team_id]
        record.append(away_team_attributes_df.values.tolist())

        match_id = row['match_api_id']
        # B365 odds
        match_B365_odds_df = B365_odds_df.loc[B365_odds_df['match_api_id'] == match_id]
        record.append(match_B365_odds_df.values.tolist())

        # BW odds
        match_BW_odds_df = BW_odds_df.loc[BW_odds_df['match_api_id'] == match_id]
        record.append(match_BW_odds_df.values.tolist())

        # IW odds
        match_IW_odds_df = IW_odds_df.loc[IW_odds_df['match_api_id'] == match_id]
        record.append(match_IW_odds_df.values.tolist())

        # LB odds
        match_LB_odds_df = LB_odds_df.loc[LB_odds_df['match_api_id'] == match_id]
        record.append(match_LB_odds_df.values.tolist())

        # Convert record (nested list) to a flat list
        flat_record = flatten_nested_list(record)

        # Remove unnecessary values ('home_team_id', 'away_team_id', 'match_id')
        flat_record = list(
            filter(lambda elem: elem != home_team_id and elem != away_team_id and elem != match_id, flat_record))

        data.append(flat_record)

    # Create the DataFrame
    MLNN_input_layer_df = pd.DataFrame(data=data, columns=column_names)

    # Add label ('result') column to DataFrame
    MLNN_input_layer_df['result'] = match_results_df['result']

    # Team_Attributes table does not contain the attributes of all the teams in the Match table
    # Remove missing values
    MLNN_feature_layer_df = MLNN_input_layer_df.dropna()

    # Export dataframe to csv
    MLNN_feature_layer_df.to_csv('MLNN_input.csv', index=False)


def flatten_nested_list(nested_list):
    # Convert a nested list to a flat list
    flatList = []

    for element in nested_list:
        if isinstance(element, list):
            flatList.extend(flatten_nested_list(element))
        else:
            flatList.append(element)

    return flatList
