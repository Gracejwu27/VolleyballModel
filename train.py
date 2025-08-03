import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import time
import torch
from preprocess import getMatchData, getData
import time

def trainModel(df_matches: pd.DataFrame, df_players: pd.DataFrame):
    """
    Trains a model on the provided data.
    The model is trained to predict the match result based on player statistics.

    Args:
        data (_type_): _description_
    """
    print("Training model...")
    
    # 1. Calculate Overall Team Win Rates from df_matches
    team_matches_played = {}
    team_wins = {}
    for index, row in df_matches.iterrows():
        winner = row['Winner']
        loser = row['Loser']
        team_matches_played[winner] = team_matches_played.get(winner, 0) + 1
        team_matches_played[loser] = team_matches_played.get(loser, 0) + 1
        team_wins[winner] = team_wins.get(winner, 0) + 1
        team_wins.setdefault(loser, 0)
    overall_win_rates = {team: team_wins[team] / team_matches_played[team] for team in team_matches_played
                         if team_matches_played[team] > 0}
    df_overall_win_rates = pd.DataFrame(overall_win_rates.items(), columns=['Team', 'OverallWinRate'])
    
    # 2. Calculate Overall Average Player Skills per Team from df_players
    skill_cols = ['attack', 'block', 'dig', 'set', 'serve', 'receive']
    df_avg_player_skills = df_players.groupby('country')[skill_cols].mean().reset_index()
    df_avg_player_skills.rename(columns={'country': 'Team'}, inplace=True)
    
    # 3. Create a combined Team Stats DataFrame
    # Includes Team, OverallWinRate, and average player skills
    df_team_stats = pd.merge(df_overall_win_rates, df_avg_player_skills, on = 'Team', how='left')
    # Impute missing player skill averages with the mean of all available players
    # This handles cases where a team might be in df_matches but not have player data in df_players
    for col in skill_cols:
        if df_team_stats[col].isnull().any():
            mean_val = df_team_stats[col].mean()
            df_team_stats[col].fillna(mean_val, inplace=True)
    df_team_stats['OverallWinRate'].fillna(0.5, inplace=True) # Default win rate to 0.5 if no matches played
    
    print("Team Stats DataFrame:", df_team_stats.head())
    
    # 4. Create Machine Learning Dataset
    ml_data = []
    for index, match in df_matches.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']

        home_stats = df_team_stats[df_team_stats['Team'] == home_team].iloc[0] if home_team in df_team_stats['Team'].values else None
        away_stats = df_team_stats[df_team_stats['Team'] == away_team].iloc[0] if away_team in df_team_stats['Team'].values else None

        # Skip match if aggregated stats for either team are not found
        if home_stats is None or away_stats is None:
            print(f"Warning: Missing aggregated stats for {home_team} or {away_team}. Skipping match.")
            continue

        # Target variable: 1 if HomeTeam wins, 0 if AwayTeam wins
        target = 1 if match['Winner'] == home_team else 0

        row_data = {
            'HomeTeam_ID': home_team, # Original team IDs for One-Hot Encoding
            'AwayTeam_ID': away_team,
            'WinRate_Diff': home_stats['OverallWinRate'] - away_stats['OverallWinRate'],
            'Attack_Diff': home_stats['attack'] - away_stats['attack'],
            'Block_Diff': home_stats['block'] - away_stats['block'],
            'Serve_Diff': home_stats['serve'] - away_stats['serve'],
            'Dig_Diff': home_stats['dig'] - away_stats['dig'],
            'Receive_Diff': home_stats['receive'] - away_stats['receive'],
            'Target': target
        }
        ml_data.append(row_data)

    df_ml = pd.DataFrame(ml_data)
    if df_ml.empty:
        print("No valid data for training the model. Exiting.")
        return None, None, None
    
    # 5. Features X and Target Variable y (y is the match result)
    X = df_ml.drop(columns=['Target'], axis = 1)
    y = df_ml['Target']
    
    numerical_features = [col for col in X.columns if 'Diff' in col]
    categorical_features = ['HomeTeam_ID', 'AwayTeam_ID']
    
    # Creating a preprocessor that applies different transformations to numerical and categorical features
    preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features), # Standardize numerical features
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # One-Hot Encode categorical team IDs
    ])
    
    # Create full model pipeline that goes from preprocessing -> model training
    
    # Create the full model pipeline: Preprocessing -> Classifier
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42)) # Using RandomForestClassifier
    ])
    
    # 6. Split data into training and testing sets
    if X.shape[0] < 2:
        print("Not enough samples to perform a train-test split (need at least 2 samples).")
        return None, None, None # Return None if too few samples
    
    test_size_val = 0.2 # Test size 20%

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42, stratify=y)

    print("\n--- Training the model ---")
    model_pipeline.fit(X_train, y_train) # Train the entire pipeline
    print("Model training complete.")

    # Return the trained pipeline, along with the test sets for evaluation
    return model_pipeline, X_test, y_test, df_team_stats

def evaluateModel(model, X_test, y_test):
    """
    Evaluates the trained model using the test set and prints the accuracy, classification report, and confusion matrix.

    Args:
        model: -
        X_test: The test features.
        y_test: The true labels for the test set.
    """
    if model is None or X_test is None or y_test is None or X_test.empty or y_test.empty:
        print("Model or test data is None or empty. Cannot evaluate.")
        return
    
    print("\n--- Evaluating the model ---")
    y_pred = model.predict(X_test) # Predict using the trained model
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def predictMatchOutcome(model, home_team, away_team, df_team_stats):
    """
    Predicts the outcome of a match between two teams using the trained model.
    Args:
        model: The trained model.
        home_team: The ID of the home team.
        away_team: The ID of the away team.
        df_team_stats: DataFrame containing team statistics.
    Returns:
        The predicted outcome (1 for home win, 0 for away win).
    """
    print(f"\nPredicting match outcome for {home_team} vs {away_team}...")
    
    # 1. Look up aggregated stats for Home and Away teams
    home_stats = df_team_stats[df_team_stats['Team'] == home_team]
    away_stats = df_team_stats[df_team_stats['Team'] == away_team]
    
    if home_stats.empty:
        print(f"Error: Stats not found for Home Team '{home_team}'. Cannot predict.")
        return None
    if away_stats.empty:
        print(f"Error: Stats not found for Away Team '{away_team}'. Cannot predict.")
        return None

    home_stats = home_stats.iloc[0] # Get the Series for the home team
    away_stats = away_stats.iloc[0] # Get the Series for the away team
    
    # 2. Prepare the input data for prediction
    # Create a DataFrame with the same structure as the training data
    prediction_data = {
        'HomeTeam_ID': [home_team], 
        'AwayTeam_ID': [away_team],
        'WinRate_Diff': [home_stats['OverallWinRate'] - away_stats['OverallWinRate']],
        'Attack_Diff': [home_stats['attack'] - away_stats['attack']],
        'Block_Diff': [home_stats['block'] - away_stats['block']],
        'Serve_Diff': [home_stats['serve'] - away_stats['serve']],
        'Dig_Diff': [home_stats['dig'] - away_stats['dig']],
        'Receive_Diff': [home_stats['receive'] - away_stats['receive']]
    }
    
    raw_feature_cols = [
        'HomeTeam_ID', 'AwayTeam_ID', 'WinRate_Diff',
        'Attack_Diff', 'Block_Diff', 'Serve_Diff', 'Dig_Diff', 'Receive_Diff'
    ]
    
    df_predict_input = pd.DataFrame(prediction_data, columns=raw_feature_cols)
    
    print("Input Data for Prediction:\n", df_predict_input)
    
    #3. Make a prediction using the model
    
    prediction = model.predict(df_predict_input)[0]
    
    if prediction == 1:
        print(f"Predicted Outcome: {home_team} wins!")
        return 1
    else:
        print(f"Predicted Outcome: {away_team} wins!")
        return 0

def main():
    """
    Main function to load data, train, and evaluate the model.
    """
    print("Loading player data...")
    # Get player data from the preprocess module
    df_players = getData() 
    if df_players is None or df_players.empty:
        print("Error: Player data (df_players) is not loaded or is empty. Cannot proceed with model training.")
        return
    print("Player data head:\n", df_players.head())

    print("\nLoading match data...")
    # Get match data from the preprocess module
    df_matches = pd.read_csv('vnl_matches_saved.csv')  
    if df_matches is None or df_matches.empty:
        print("Error: Match data (df_matches) is not loaded or is empty. Cannot proceed with model training.")
        return
    print("Match data head:\n", df_matches.head())

    # Train the model (it returns the trained pipeline and test data)
    trained_model, X_test, y_test, df_team_stats_for_pred = trainModel(df_matches, df_players)

    # Evaluate the model using the returned test data
    evaluateModel(trained_model, X_test, y_test)

    if trained_model:
        print("\nModel training and evaluation process completed successfully!")
        home_team_to_predict = 'USA'
        away_team_to_predict = 'ITA'
        
        if home_team_to_predict not in df_team_stats_for_pred['Team'].values:
            print(f"Error: Team '{home_team_to_predict}' not found in team statistics. Cannot predict.")
            return
        if away_team_to_predict not in df_team_stats_for_pred['Team'].values:
            print(f"Error: Team '{away_team_to_predict}' not found in team statistics. Cannot predict.")
            return
            
        predictMatchOutcome(trained_model, home_team_to_predict, away_team_to_predict, df_team_stats_for_pred)
 
    else:
        print("\nModel training failed or was skipped due to insufficient data or errors during training.")


if __name__ == "__main__":
    main()
