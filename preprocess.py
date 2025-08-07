import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import time
import os
# Import Selenium components
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

def safe_int_conversion(text):
    """Converts a string to an int, returning None if conversion fails (e.g., on '-')."""
    try:
        return int(text)
    except ValueError:
        return None # Or np.nan if you prefer pandas' NaN for numerical columns

def calculate_total_points(row, team_prefix):
    """
    Calculates the total points for a team in a match based on the set scores.
    """
    total = 0
    for i in range(1, 6):
        score = row.get(f'Set{i}_{team_prefix}Score')
        if score is not None:
            total += score
        else: # Stop if a set score is None, implies match ended
            break
    return total

def getMatchData():
    """
    Gets data from an HTML page and returns it as a pandas DataFrame.
    Should be used to scrape the data from the HTML page.
    Should be getting the Match ID, Team 1, Team 2, and Match Result.
    """
 # --- Configuration ---
    BASE_URL = "https://vnlw.volleystation.com"
    SCHEDULE_URLS = [
        f"{BASE_URL}/en/season/74/phase-623-wvnl2023/results/", # VNL 2023 Women
        f"{BASE_URL}/en/season/102/phase-1784-vnl-2024-women/results/", # VNL 2024 Women
        f"{BASE_URL}/en/phase-3451-vnl-2025-women/results/" # VNL 2025 Women
    ]

    # --- Selenium Setup ---
    CHROMEDRIVER_PATH = '/Users/gracewu/VolleyML/chromedriver'

    options = webdriver.ChromeOptions()
    options.add_argument('--headless') # Run Chrome in headless mode (no browser window)
    options.add_argument('--no-sandbox') # Required for some environments (e.g., Docker, some Linux systems)
    options.add_argument('--disable-dev-shm-usage') # Required for some environments
    options.add_argument('window-size=1920x1080') # Set a consistent window size
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36') # Set User-Agent

    driver = None # Initialize driver to None for proper cleanup in finally block
    all_matches_data = []
    try:
        service = Service(executable_path=CHROMEDRIVER_PATH)
        driver = webdriver.Chrome(service=service, options=options)

        # Loop through each URL
        for url in SCHEDULE_URLS:
            print(f"Getting match data from URL: {url}...")
            
            # Navigate to the current URL
            driver.get(url)

            # Wait for the main content to load. Adjust the condition as needed.
            wait = WebDriverWait(driver, 20)
            try:
                matches_list_container_element = wait.until(
                    EC.presence_of_element_located((By.CLASS_NAME, 'matches-list-container'))
                )
            except TimeoutException:
                print(f"Warning: Timed out waiting for matches on {url}. Skipping this URL.")
                continue # Skip to the next URL in the list if the element is not found

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            matches_list_container = soup.find('div', class_='matches-list-container')
            
            if not matches_list_container:
                print(f"Warning: No matches found in the HTML structure for {url}. Skipping this URL.")
                continue

            day_groups = matches_list_container.find_all('div', class_='day-group')
            for day_group in day_groups:
                day_label = day_group.find('div', class_='day-label')
                current_date = day_label.text.strip() if day_label else "Unknown Date"
                print(f"Processing matches for date: {current_date}")
                
                match_containers = day_group.find_all('a', href=lambda href: href and '/matches/' in href)
                for match_container in match_containers:
                    match_detail_url = f"{BASE_URL}{match_container['href']}"

                    try:
                        teams_div = match_container.find('div', class_ = 'details').find('div', class_='teams')
                        teams = teams_div.find_all('div', class_='team')

                        home_team = teams[0].find('div', class_='short-name').text.strip()
                        away_team = teams[1].find('div', class_='short-name').text.strip()
                        
                        match_score_div = match_container.find('div', class_='match-score')
                        home_set_score = None
                        away_set_score = None
                        if match_score_div:
                            set_scores_overall = match_score_div.find_all('div', class_='set-score')
                            if len(set_scores_overall) >= 2:
                                home_set_score = safe_int_conversion(set_scores_overall[0].text.strip())
                                away_set_score = safe_int_conversion(set_scores_overall[1].text.strip())
                            else:
                                print(f"Warning: Not enough overall set scores found for {home_team} vs {away_team} in {match_detail_url}. Skipping overall set scores.")
                        else:
                            print(f"Warning: No match score div found for {home_team} vs {away_team} at {current_date}. Skipping overall set scores.")
                        
                        sets_score_container = match_container.find('div', class_='sets-score')
                        set_results = sets_score_container.find_all('div', class_='result')

                        set_scores_dict = {f'Set{j}_HomeScore': None for j in range(1, 6)}
                        set_scores_dict.update({f'Set{j}_AwayScore': None for j in range(1, 6)})

                        for set_num, set_result_div in enumerate(set_results):
                            scores_in_set = set_result_div.find_all('div', class_='set-score')

                            if len(scores_in_set) == 2:
                                set_home_score = safe_int_conversion(scores_in_set[0].text.strip())
                                set_away_score = safe_int_conversion(scores_in_set[1].text.strip())

                                if set_num < 5:
                                    set_scores_dict[f'Set{set_num+1}_HomeScore'] = set_home_score
                                    set_scores_dict[f'Set{set_num+1}_AwayScore'] = set_away_score
                            else:
                                print(f"Warning: Unexpected number of set scores in result div for set {set_num+1} for {match_detail_url}. Skipping set scores.")
                        
                        match_data = {
                            'Date': current_date,
                            'HomeTeam': home_team,
                            'AwayTeam': away_team,
                            'HomeSetsWon': home_set_score,
                            'AwaySetsWon': away_set_score,
                            'MatchDetailURL': match_detail_url,
                            **set_scores_dict
                        }
                        
                        match_data['HomeTotalPoints'] = calculate_total_points(match_data, 'Home')
                        match_data['AwayTotalPoints'] = calculate_total_points(match_data, 'Away')

                        all_matches_data.append(match_data)
                                    
                    except AttributeError as e:
                        print(f"Skipping match (possible incomplete data or structure change): {e} for URL {match_detail_url}")
                        continue
                    except ValueError as e:
                        print(f"Error converting data for {home_team} vs {away_team}: {e} for URL {match_detail_url}")
                        continue
            
    except TimeoutException:
        print(f"Timed out waiting for page elements to load at {SCHEDULE_URLS[0]}. The website might be slow or heavily dynamic.")
    except WebDriverException as e:
        print(f"A WebDriver error occurred during navigation or element location: {e}")
        print("Please ensure ChromeDriver is installed and its path is correct, and that its version matches your Chrome browser.")
    finally:
        if driver:
            driver.quit()
    
    print(f"\nSuccessfully scraped {len(all_matches_data)} matches.")
    
    df_matches = pd.DataFrame(all_matches_data)
    
    if not df_matches.empty and 'HomeSetsWon' in df_matches.columns and 'AwaySetsWon' in df_matches.columns:
        df_matches['Winner'] = df_matches.apply(
            lambda row: row['HomeTeam'] if row['HomeSetsWon'] is not None and row['AwaySetsWon'] is not None and row['HomeSetsWon'] > row['AwaySetsWon'] else (
                row['AwayTeam'] if row['HomeSetsWon'] is not None and row['AwaySetsWon'] is not None and row['AwaySetsWon'] > row['HomeSetsWon'] else None
            ), axis=1
        )
        df_matches['Loser'] = df_matches.apply(
            lambda row: row['AwayTeam'] if row['HomeSetsWon'] is not None and row['AwaySetsWon'] is not None and row['HomeSetsWon'] > row['AwaySetsWon'] else (
                row['HomeTeam'] if row['HomeSetsWon'] is not None and row['AwaySetsWon'] is not None and row['AwaySetsWon'] > row['HomeSetsWon'] else None
            ), axis=1
        )
    else:
        df_matches['Winner'] = None
        df_matches['Loser'] = None
    
    return df_matches

def getData():
    """
    Loads in VNL data set from a CSV file and returns it as a pandas DataFrame.
    Data includes:
    player,country,age,attack,block,serve,set,dig,receive,position
    as averages over the whole tournament (could also scrape data for each match)
    """
    df = pd.read_csv('vnl.csv')
    if df.empty:
        print("DataFrame is empty. Please check the CSV file.")
        return pd.DataFrame()
    print("Data loaded successfully.")
    return df

# Define filenames for your saved data
MATCH_DATA_FILE = 'vnl_matches_saved.csv'
PLAYER_DATA_FILE = 'vnl_2024_players_saved.csv' 

def saveMatchData():
    df_matches = pd.DataFrame() # Initialize as empty DataFrame
    if os.path.exists(MATCH_DATA_FILE):
        print(f"Loading match data from {MATCH_DATA_FILE}...")
        try:
            df_matches = pd.read_csv(MATCH_DATA_FILE)
            print("Match data loaded successfully.")
        except Exception as e:
            print(f"Error loading match data from CSV: {e}. Attempting to fetch live data.")
            df_matches = getMatchData()
            if not df_matches.empty:
                df_matches.to_csv(MATCH_DATA_FILE, index=False)
                print(f"Match data fetched and saved to {MATCH_DATA_FILE}.")
    else:
        print("Match data CSV not found. Fetching live data...")
        df_matches = getMatchData()
        if not df_matches.empty:
            df_matches.to_csv(MATCH_DATA_FILE, index=False)
            print(f"Match data fetched and saved to {MATCH_DATA_FILE}.")

    # --- Get player statistics data ---
    df_players = pd.DataFrame() # Initialize as empty DataFrame
    if os.path.exists(PLAYER_DATA_FILE):
        print(f"Loading player statistics data from {PLAYER_DATA_FILE}...")
        try:
            df_players = pd.read_csv(PLAYER_DATA_FILE)
            print("Player statistics data loaded successfully.")
        except Exception as e:
            print(f"Error loading player data from CSV: {e}. Attempting to fetch live data (if getData is a scraper).")
            print("Please ensure your player data file exists and is correctly loaded by getData().")
            return # Exit if player data isn't available

    else:
        print("Player statistics data CSV not found. Fetching live data (if getData is a scraper)...")
        print("Please ensure your player data file exists and is correctly loaded by getData().")
        return # Exit if player data isn't available

    # --- Proceed with your visualizations or analysis ---
    print("\n--- DataFrames ready for analysis ---")
    print("df_matches head:\n", df_matches.head())
    print("\ndf_players head:\n", df_players.head())

def main():
    """
    Main function to scrape data
    """
    getMatchData()
    saveMatchData()
    
    
if __name__ == "__main__":
    main()