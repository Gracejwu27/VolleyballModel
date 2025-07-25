import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

# --- Configuration for saving plots ---
OUTPUT_FOLDER = 'dataVisualizations'
# Ensure the output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"Created directory: {OUTPUT_FOLDER}")
else:
    print(f"Directory already exists: {OUTPUT_FOLDER}")


# Set Seaborn style once globally for consistency
sns.set_style("whitegrid")

# --- Basic Visualization Functions (from your original code) ---

def plot_top_winning_teams(df_matches: pd.DataFrame):
    """
    Plots the top 10 winning teams from the match data and saves the plot.
    """
    print("\n--- Plotting Top 10 Winning Teams ---")
    plt.figure(figsize=(10, 6))
    top_winners = df_matches['Winner'].value_counts().head(10)
    sns.barplot(x=top_winners.index, y=top_winners.values, palette="viridis", hue=top_winners.index, legend=False)
    plt.title('Top 10 Winning Teams in VNL 2024')
    plt.xlabel('Country')
    plt.ylabel('Number of Wins')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'top_10_winning_teams.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_match_lengths(df_matches: pd.DataFrame):
    """
    Plots the distribution of match lengths (total sets played) and saves the plot.
    """
    if 'TotalSets' not in df_matches.columns:
        df_matches['TotalSets'] = df_matches['HomeSetsWon'] + df_matches['AwaySetsWon']

    print("\n--- Plotting Distribution of Match Lengths (in Sets) ---")
    plt.figure(figsize=(8, 5))
    sns.countplot(x='TotalSets', data=df_matches, palette="magma", hue='TotalSets', legend=False)
    plt.title('Distribution of Match Lengths (Total Sets)')
    plt.xlabel('Total Sets Played')
    plt.ylabel('Number of Matches')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'match_lengths_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_top_attack_players(df_players: pd.DataFrame):
    """
    Plots the top 10 players by average attack points and saves the plot.
    """
    print("\n--- Plotting Top 10 Players by Attack Points ---")
    plt.figure(figsize=(12, 7))
    top_attackers = df_players.sort_values(by='attack', ascending=False).head(10)
    sns.barplot(x='player', y='attack', data=top_attackers, palette="rocket", hue='player', legend=False)
    plt.title('Top 10 Players by Average Attack Points')
    plt.xlabel('Player Name')
    plt.ylabel('Average Attack Points')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'top_10_attack_players.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_attack_by_position(df_players: pd.DataFrame):
    """
    Plots the distribution of average attack points by player position using box plots and saves the plot.
    """
    print("\n--- Plotting Average Attack Points by Player Position ---")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='position', y='attack', data=df_players, palette="pastel")
    plt.title('Distribution of Average Attack Points by Player Position')
    plt.xlabel('Position')
    plt.ylabel('Average Attack Points')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'attack_by_position.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_age_distribution(df_players: pd.DataFrame):
    """
    Plots the age distribution of players using a histogram and saves the plot.
    """
    print("\n--- Plotting Age Distribution of Players ---")
    plt.figure(figsize=(8, 5))
    sns.histplot(df_players['age'], bins=range(18, 35, 1), kde=True, color='skyblue')
    plt.title('Age Distribution of Players')
    plt.xlabel('Age')
    plt.ylabel('Number of Players')
    plt.xticks(range(18, 35, 2))
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'age_distribution.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# --- Advanced Visualization Functions ---

def plot_player_skill_archetypes(df_players: pd.DataFrame):
    """
    Plots a scatter plot showing player skill archetypes based on Attack vs. Block points,
    colored by player position and sized by age, and saves the plot.
    """
    print("\n--- Plotting Player Skill Archetypes (Attack vs. Block) ---")
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='attack',
        y='block',
        data=df_players,
        hue='position',
        size='age',
        sizes=(50, 500),
        alpha=0.7,
        palette='deep'
    )
    plt.title('Player Skill Archetypes: Attack vs. Block Points by Position')
    plt.xlabel('Average Attack Points')
    plt.ylabel('Average Block Points')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Position', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'player_skill_archetypes.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_team_offensive_defensive_balance(df_matches: pd.DataFrame):
    """
    Plots a scatter plot showing each team's offensive (Points Scored) vs.
    defensive (Points Conceded) balance, and saves the plot.
    """
    print("\n--- Plotting Team Offensive/Defensive Balance ---")

    team_points_summary = {}
    for index, row in df_matches.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        home_points_scored = row['HomeTotalPoints']
        away_points_scored = row['AwayTotalPoints']

        team_points_summary.setdefault(home_team, {'PointsScored': 0, 'PointsConceded': 0})
        if not pd.isna(home_points_scored):
            team_points_summary[home_team]['PointsScored'] += home_points_scored
        if not pd.isna(away_points_scored):
            team_points_summary[home_team]['PointsConceded'] += away_points_scored

        team_points_summary.setdefault(away_team, {'PointsScored': 0, 'PointsConceded': 0})
        if not pd.isna(away_points_scored):
            team_points_summary[away_team]['PointsScored'] += away_points_scored
        if not pd.isna(home_points_scored):
            team_points_summary[away_team]['PointsConceded'] += home_points_scored

    df_team_points = pd.DataFrame.from_dict(team_points_summary, orient='index')
    df_team_points.index.name = 'Team'
    df_team_points = df_team_points.reset_index()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PointsScored',
        y='PointsConceded',
        data=df_team_points,
        hue='Team',
        size='PointsScored',
        sizes=(100, 1000),
        alpha=0.7,
        palette='tab10'
    )
    plt.title('Team Offensive vs. Defensive Balance (Total Points in Matches)')
    plt.xlabel('Total Points Scored')
    plt.ylabel('Total Points Conceded')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'team_offense_defense_balance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_team_win_rate_vs_attack_power(df_matches: pd.DataFrame, df_players: pd.DataFrame):
    """
    Plots a scatter plot showing the relationship between a team's Win Rate
    and its Average Player Attack Power, and saves the plot.
    """
    print("\n--- Plotting Team Win Rate vs. Average Team Attack Power ---")

    team_matches_played = {}
    team_wins = {}

    for index, row in df_matches.iterrows():
        winner = row['Winner']
        loser = row['Loser']

        team_matches_played.setdefault(winner, 0)
        team_matches_played.setdefault(loser, 0)
        team_wins.setdefault(winner, 0)
        team_wins.setdefault(loser, 0)

        team_matches_played[winner] += 1
        team_matches_played[loser] += 1
        team_wins[winner] += 1

    win_rates = {team: team_wins[team] / team_matches_played[team] for team in team_matches_played if team_matches_played[team] > 0}
    df_win_rates = pd.DataFrame(win_rates.items(), columns=['Team', 'WinRate'])

    df_team_attack_avg = df_players.groupby('country')['attack'].mean().reset_index()
    df_team_attack_avg.columns = ['Team', 'AvgPlayerAttack']

    df_combined_team_stats = pd.merge(df_win_rates, df_team_attack_avg, on='Team', how='outer')

    df_combined_team_stats.fillna({'WinRate': 0}, inplace=True)
    if not df_combined_team_stats['AvgPlayerAttack'].empty:
        df_combined_team_stats['AvgPlayerAttack'].fillna(df_combined_team_stats['AvgPlayerAttack'].mean(), inplace=True)
    else:
        df_combined_team_stats['AvgPlayerAttack'].fillna(0, inplace=True)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='AvgPlayerAttack',
        y='WinRate',
        data=df_combined_team_stats,
        hue='Team',
        size='WinRate',
        sizes=(100, 1000),
        alpha=0.8,
        palette='Spectral'
    )
    for i, row in df_combined_team_stats.iterrows():
        plt.text(row['AvgPlayerAttack'] * 1.01, row['WinRate'] * 1.01, row['Team'],
                 fontsize=9, ha='left', va='bottom')

    plt.title('Team Win Rate vs. Average Player Attack Power')
    plt.xlabel('Average Attack Points of All Players on Team')
    plt.ylabel('Team Win Rate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'team_win_rate_vs_attack_power.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_team_win_rate_vs_skill_power(df_matches: pd.DataFrame, df_players: pd.DataFrame, skill_col: str, skill_label: str, filename_suffix: str, palette_name: str = 'Spectral'):
    """
    Generalized function to plot team Win Rate vs. Average Player Skill Power for a given skill.

    Args:
        df_matches: DataFrame containing match results.
        df_players: DataFrame containing player statistics.
        skill_col: The name of the player skill column to use (e.g., 'block', 'serve', 'dig', 'receive').
        skill_label: A human-readable label for the skill (e.g., 'Block', 'Serve', 'Dig', 'Receive').
        filename_suffix: Suffix for the saved plot filename (e.g., 'block_power').
        palette_name: Seaborn palette to use for the scatter plot.
    """
    print(f"\n--- Plotting Team Win Rate vs. Average Player {skill_label} Power ---")

    # Step 1: Calculate Team Win Rates from df_matches (common for all these plots)
    team_matches_played = {}
    team_wins = {}

    for index, row in df_matches.iterrows():
        winner = row['Winner']
        loser = row['Loser']

        team_matches_played.setdefault(winner, 0)
        team_matches_played.setdefault(loser, 0)
        team_wins.setdefault(winner, 0)
        team_wins.setdefault(loser, 0)

        team_matches_played[winner] += 1
        team_matches_played[loser] += 1
        team_wins[winner] += 1

    win_rates = {team: team_wins[team] / team_matches_played[team] for team in team_matches_played if team_matches_played[team] > 0}
    df_win_rates = pd.DataFrame(win_rates.items(), columns=['Team', 'WinRate'])

    # Step 2: Calculate Average Team Skill Power from df_players for the specified skill
    df_team_skill_avg = df_players.groupby('country')[skill_col].mean().reset_index()
    df_team_skill_avg.columns = ['Team', 'AvgPlayerSkill']

    # Step 3: Merge these two DataFrames
    df_combined_team_stats = pd.merge(df_win_rates, df_team_skill_avg, on='Team', how='outer')

    # Handle NaNs: Fill WinRate with 0 for teams with no wins, fill AvgPlayerSkill with overall mean
    df_combined_team_stats.fillna({'WinRate': 0}, inplace=True)
    if not df_combined_team_stats['AvgPlayerSkill'].empty:
        df_combined_team_stats['AvgPlayerSkill'].fillna(df_combined_team_stats['AvgPlayerSkill'].mean(), inplace=True)
    else:
        df_combined_team_stats['AvgPlayerSkill'].fillna(0, inplace=True)

    # Step 4: Visualize the combined data
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='AvgPlayerSkill',
        y='WinRate',
        data=df_combined_team_stats,
        hue='Team',
        size='WinRate',
        sizes=(100, 1000),
        alpha=0.8,
        palette=palette_name
    )
    for i, row in df_combined_team_stats.iterrows():
        plt.text(row['AvgPlayerSkill'] * 1.01, row['WinRate'] * 1.01, row['Team'],
                 fontsize=9, ha='left', va='bottom')

    plt.title(f'Team Win Rate vs. Average Player {skill_label} Power')
    plt.xlabel(f'Average {skill_label} Points of All Players on Team')
    plt.ylabel('Team Win Rate')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Team', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_FOLDER, f'team_win_rate_vs_{filename_suffix}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
# --- Main Visualization Runner Function ---

def run_all_visualizations(df_matches: pd.DataFrame, df_players: pd.DataFrame):
    """
    Executes all defined basic and advanced visualization functions.
    """
    # Basic Plots
    """    plot_top_winning_teams(df_matches)
    plot_match_lengths(df_matches)
    plot_top_attack_players(df_players)
    plot_attack_by_position(df_players)
    plot_age_distribution(df_players)"""

    # Advanced Plots
    plot_player_skill_archetypes(df_players)
    plot_team_offensive_defensive_balance(df_matches)
    plot_team_win_rate_vs_skill_power(df_matches, df_players, 'attack', 'Attack', 'attack_power', 'Spectral') # Re-use for attack
    plot_team_win_rate_vs_skill_power(df_matches, df_players, 'block', 'Block', 'block_power', 'Spectral')
    plot_team_win_rate_vs_skill_power(df_matches, df_players, 'serve', 'Serve', 'serve_power', 'viridis')
    plot_team_win_rate_vs_skill_power(df_matches, df_players, 'dig', 'Dig', 'dig_power', 'plasma')
    plot_team_win_rate_vs_skill_power(df_matches, df_players, 'receive', 'Receive', 'receive_power', 'cividis')

    print("\n--- All visualizations completed! ---")


# --- Main Execution Flow (as per your request, loading data directly here) ---

def visualizeData():
    """
    Loads data from CSVs and runs all visualization functions.
    """
    print("Loading match data from 'vnl_2024_matches_saved.csv'...")
    try:
        df_matches = pd.read_csv('vnl_2024_matches_saved.csv')
    except FileNotFoundError:
        print("Error: 'vnl_2024_matches_saved.csv' not found. Please ensure the file exists in the same directory.")
        return
    except Exception as e:
        print(f"Error loading 'vnl_2024_matches_saved.csv': {e}")
        return

    print("Loading player data from 'vnl.csv'...")
    try:
        df_players = pd.read_csv('vnl.csv')
    except FileNotFoundError:
        print("Error: 'vnl.csv' not found. Please ensure the file exists in the same directory.")
        return
    except Exception as e:
        print(f"Error loading 'vnl.csv': {e}")
        return
    
    # You might want to add checks here for empty DataFrames if your CSVs could be empty
    if df_matches.empty:
        print("Error: 'vnl_2024_matches_saved.csv' is empty or could not be loaded into a DataFrame.")
        return
    if df_players.empty:
        print("Error: 'vnl.csv' is empty or could not be loaded into a DataFrame.")
        return
        
    run_all_visualizations(df_matches, df_players)

def main():
    """
    Main function to initiate the visualization process.
    """
    visualizeData()

if __name__ == "__main__":
    main()