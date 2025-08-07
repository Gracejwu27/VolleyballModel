import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import time
import torch
from preprocess import getMatchData, getData
import time

def print_grid_search_results(grid_search):
    """
    Prints a formatted DataFrame of all GridSearchCV results.
    """
    print("\n--- All GridSearchCV Results ---")
    results = pd.DataFrame(grid_search.cv_results_)
    # Select and format key columns for a clean chart
    results_df = results[['param_classifier__n_estimators', 'param_classifier__max_depth',
                          'param_classifier__min_samples_leaf', 'mean_test_score', 'rank_test_score']]
    results_df.columns = ['n_estimators', 'max_depth', 'min_samples_leaf', 'mean_accuracy', 'rank']
    results_df.sort_values(by='rank', inplace=True)
    results_df.reset_index(drop=True, inplace=True)
    print(results_df)
    
def tuneRandomForestHyperparameters(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Uses GridSearchCV to find the best hyperparameters for the RandomForestClassifier.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        
    Returns:
        The best hyperparameters found as a dictionary.
    """
    print("\n--- Tuning RandomForest Hyperparameters with GridSearchCV ---")
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [5, 10, None],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    numerical_features = [col for col in X_train.columns if 'Diff' in col]
    categorical_features = ['HomeTeam_ID', 'AwayTeam_ID']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    grid_search = GridSearchCV(
        estimator=pipeline, 
        param_grid=param_grid, 
        cv=5, # 5-fold cross-validation
        scoring='accuracy', 
        n_jobs=-1, # Use all available cores
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("\nBest hyperparameters found by GridSearchCV:")
    print(grid_search.best_params_)
    
    print_grid_search_results(grid_search)
    
    best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
    return best_params

def trainModel(X: pd.DataFrame, y: pd.Series, model_type='RandomForest', params=None):
    """
    Trains a machine learning model based on the specified type and parameters.
    
    Args:
        X: The feature DataFrame (full dataset).
        y: The target Series (full dataset).
        model_type: 'RandomForest' or 'XGBoost'.
        params: Dictionary of hyperparameters for the classifier.
        
    Returns:
        A trained scikit-learn pipeline. Returns None if training fails.
    """
    print(f"\n--- Training the {model_type} model on the full dataset ---")

    if model_type == 'XGBoost':
        classifier = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', **(params or {}))
    else: # Default to RandomForest
        classifier = RandomForestClassifier(random_state=42, **(params or {}))

    numerical_features = [col for col in X.columns if 'Diff' in col]
    categorical_features = ['HomeTeam_ID', 'AwayTeam_ID']
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    model_pipeline.fit(X, y) # Train on the full dataset
    print("Model training complete.")

    return model_pipeline

def evaluateModel(model_pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Evaluates the trained model using the test set.
    """
    if model_pipeline is None or X_test is None or y_test is None or X_test.empty:
        print("Evaluation skipped.")
        return
    y_pred = model_pipeline.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predictMatchOutcome(model_pipeline: Pipeline, home_team_id: str, away_team_id: str, df_team_stats: pd.DataFrame):
    """
    Predicts the outcome of a match between two teams using the trained model.
    """
    print(f"\n--- Predicting outcome for {home_team_id} (Home) vs {away_team_id} (Away) ---")
    home_stats = df_team_stats[df_team_stats['Team'] == home_team_id]
    away_stats = df_team_stats[df_team_stats['Team'] == away_team_id]
    if home_stats.empty:
        print(f"Error: Stats not found for Home Team '{home_team_id}'. Cannot predict.")
        return None
    if away_stats.empty:
        print(f"Error: Stats not found for Away Team '{away_team_id}'. Cannot predict.")
        return None
    home_stats, away_stats = home_stats.iloc[0], away_stats.iloc[0]
    prediction_data = {
        'HomeTeam_ID': [home_team_id],
        'AwayTeam_ID': [away_team_id],
        'WinRate_Diff': [home_stats['OverallWinRate'] - away_stats['OverallWinRate']],
        'Attack_Diff': [home_stats['attack'] - away_stats['attack']],
        'Block_Diff': [home_stats['block'] - away_stats['block']],
        'Serve_Diff': [home_stats['serve'] - away_stats['serve']],
        'Dig_Diff': [home_stats['dig'] - away_stats['dig']],
        'Receive_Diff': [home_stats['receive'] - away_stats['receive']]
    }
    raw_feature_cols = [
        'HomeTeam_ID', 'AwayTeam_ID', 'WinRate_Diff', 'Attack_Diff',
        'Block_Diff', 'Serve_Diff', 'Dig_Diff', 'Receive_Diff'
    ]
    df_predict_input = pd.DataFrame(prediction_data, columns=raw_feature_cols)
    print("Prediction input features (raw):\n", df_predict_input)
    predicted_outcome = model_pipeline.predict(df_predict_input)[0]
    if predicted_outcome == 1:
        print(f"Predicted Outcome: {home_team_id} (Home) Wins!")
        return 1
    else:
        print(f"Predicted Outcome: {away_team_id} (Away) Wins!")
        return 0

# --- Main execution flow ---
def main():
    print("Loading player data...")
    df_players = getData() 
    if df_players is None or df_players.empty:
        print("Error: Player data (df_players) is not loaded or is empty. Cannot proceed.")
        return
    print("Player data head:\n", df_players.head())

    print("\nLoading match data...")
    df_matches = pd.read_csv('vnl_matches_saved.csv')  
    if df_matches is None or df_matches.empty:
        print("Error: Match data (df_matches) is not loaded or is empty. Cannot proceed.")
        return
    print("Match data head:\n", df_matches.head())
    
    # --- Prepare the dataset for training ---
    ml_data = []
    team_matches_played = {}
    team_wins = {}
    # Calculate overall win rates and team stats
    for _, row in df_matches.iterrows():
        winner, loser = row['Winner'], row['Loser']
        team_matches_played.setdefault(winner, 0); team_matches_played.setdefault(loser, 0)
        team_wins.setdefault(winner, 0); team_wins.setdefault(loser, 0)
        team_matches_played[winner] += 1; team_matches_played[loser] += 1
        team_wins[winner] += 1
    overall_win_rates = {t: team_wins[t] / team_matches_played[t] for t in team_matches_played if team_matches_played[t] > 0}
    df_overall_win_rates = pd.DataFrame(overall_win_rates.items(), columns=['Team', 'OverallWinRate'])
    
    # Calculate average player skills per team
    # Create a DataFrame for aggregated team stats
    skill_cols = ['attack', 'block', 'serve', 'dig', 'receive']
    df_avg_player_skills = df_players.groupby('country')[skill_cols].mean().reset_index()
    df_avg_player_skills.rename(columns={'country': 'Team'}, inplace=True)
    df_team_stats = pd.merge(df_overall_win_rates, df_avg_player_skills, on='Team', how='left')
    # Fill missing values in team stats
    for col in skill_cols:
        if df_team_stats[col].isnull().any():
            df_team_stats[col].fillna(df_team_stats[col].mean(), inplace=True)
    df_team_stats['OverallWinRate'].fillna(0.5, inplace=True)
    # Build the dataset for each match
    for _, match in df_matches.iterrows():
        home_team, away_team = match['HomeTeam'], match['AwayTeam']
        home_stats = df_team_stats[df_team_stats['Team'] == home_team].iloc[0] if home_team in df_team_stats['Team'].values else None
        away_stats = df_team_stats[df_team_stats['Team'] == away_team].iloc[0] if away_team in df_team_stats['Team'].values else None
        if home_stats is None or away_stats is None: continue
        target = 1 if match['Winner'] == home_team else 0
        row_data = {
            'HomeTeam_ID': home_team, 'AwayTeam_ID': away_team,
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
        print("Error: No data prepared for model.")
        return
    X = df_ml.drop('Target', axis=1)
    y = df_ml['Target']

    # --- Step 1: Tune Random Forest Hyperparameters ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    best_rf_params = tuneRandomForestHyperparameters(X_train, y_train)

    # --- Step 2: Train and Evaluate Models with and without tuning ---
    print("\n### Evaluating Random Forest with Default Parameters ###")
    rf_model_default = trainModel(X, y, model_type='RandomForest')
    evaluateModel(rf_model_default, X_test, y_test)
    
    print("\n### Evaluating Random Forest with Tuned Parameters ###")
    rf_model_tuned = trainModel(X, y, model_type='RandomForest', params=best_rf_params)
    evaluateModel(rf_model_tuned, X_test, y_test)

    print("\n### Evaluating XGBoost Classifier ###")
    xgb_model = trainModel(X, y, model_type='XGBoost')
    evaluateModel(xgb_model, X_test, y_test)

    # Use the best model for prediction (the tuned Random Forest)
    trained_model = xgb_model
    
    if trained_model:
        print("\nModel training and evaluation process completed successfully!")
        home_team_to_predict = 'BRA'
        away_team_to_predict = 'ITA'
        
        if home_team_to_predict not in df_team_stats['Team'].values:
            print(f"Error: Team '{home_team_to_predict}' not found in team statistics. Cannot predict.")
            return
        if away_team_to_predict not in df_team_stats['Team'].values:
            print(f"Error: Team '{away_team_to_predict}' not found in team statistics. Cannot predict.")
            return
            
        predictMatchOutcome(trained_model, home_team_to_predict, away_team_to_predict, df_team_stats)
 
    else:
        print("\nModel training failed or was skipped due to insufficient data or errors during training.")

if __name__ == "__main__":
    main()