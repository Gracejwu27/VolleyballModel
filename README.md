# VolleyML: Volleyball Match Winner Prediction

## 📖 Project Overview

VolleyML is a machine learning project designed to predict the winner of women's Volleyball Nations League (VNL) matches. It works by scraping live match and player data, processing it to create meaningful features, and then training a Random Forest model to predict outcomes. The project also includes a suite of data visualizations to analyze team and player performance.

---

## ✨ Key Features

*   **🌐 Web Scraping**: Automatically scrapes match results and schedules from the official VNL website using Selenium and BeautifulSoup.
*   **🔧 Data Preprocessing**: Cleans and merges separate datasets for match results and player statistics into a unified, model-ready format.
*   **🤖 Machine Learning Model**: Implements a `RandomForestClassifier` within a scikit-learn `Pipeline` to predict match winners based on the statistical differences between the two competing teams.
*   **📊 Model Evaluation**: Provides a complete evaluation of the model's performance, including accuracy, a classification report, and a confusion matrix.
*   **🚀 Live Prediction**: Includes a function to predict the outcome of a hypothetical match between any two teams with available data.
*   **📈 Data Visualization**: Generates and saves a variety of plots to explore the data, such as comparing team win rates against their average skill power.

---

## 📂 File Structure

```
VolleyML/
├── dataVisualizations/         # Output folder for generated plots
│   ├── team_win_rate_vs_attack_power.png
│   └── ...
├── preprocess.py               # Handles data scraping and loading
├── train.py                    # Main script for model training, evaluation, and prediction
├── visualizeData.py            # Generates and saves data visualizations
├── vnl.csv                     # Static CSV with player statistics (Required)
├── vnl_2024_matches_saved.csv  # Cached data from scraping
└── chromedriver                # Selenium WebDriver for Chrome (Required)
```

---

## ⚙️ Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repository-url>
    cd VolleyML
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Create a `requirements.txt` file with the following content:
    ```
    pandas
    scikit-learn
    selenium
    beautifulsoup4
    matplotlib
    seaborn
    torch
    numpy
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

4.  **ChromeDriver**
    *   Download the `chromedriver` executable that matches your version of Google Chrome.
    *   Place it in the root directory of the project.
    *   Ensure the path in [`preprocess.py`](preprocess.py) is correct: `CHROMEDRIVER_PATH = '/path/to/your/VolleyML/chromedriver'`

5.  **Player Data**
    *   Ensure you have the player statistics file named `vnl.csv` in the root directory. This file is loaded by the `getData()` function in [`preprocess.py`](preprocess.py).

---

## 🚀 How to Run

The project is designed to be run in sequence.

1.  **Scrape and Prepare Data**
    First, run the preprocessing script. This will scrape the latest match data from the web and save it to `vnl_2024_matches_saved.csv` to avoid re-scraping every time.
    ```bash
    python3 preprocess.py
    ```

2.  **Generate Visualizations** (Optional)
    To explore the data and see relationships between team stats and performance, run the visualization script. The plots will be saved in the `dataVisualizations/` folder.
    ```bash
    python3 visualizeData.py
    ```

3.  **Train and Evaluate the Model**
    This is the main script. It will load the preprocessed data, train the model, print an evaluation report, and run a sample prediction for a hardcoded match (e.g., ITA vs. BRA).
    ```bash
    python3 train.py
    ```

---

## 🧠 Model Details

The prediction model is built on the idea that the difference in skill between two teams is a strong predictor of the outcome.

*   **Features**: The model doesn't use raw team stats. Instead, its features are the *differences* between the home and away teams for key metrics:
    *   `WinRate_Diff`
    *   `Attack_Diff`
    *   `Block_Diff`
    *   `Serve_Diff`
    *   `Dig_Diff`
    *   `Receive_Diff`
*   **Preprocessing**: A `ColumnTransformer` within a `Pipeline` handles all preprocessing automatically:
    *   **`StandardScaler`**: Applied to all numerical `_Diff` features to normalize their scale.
    *   **`OneHotEncoder`**: Applied to the `HomeTeam_ID` and `AwayTeam_ID` to convert team names into a numerical format the model can understand.
*   **Algorithm**: A `RandomForestClassifier` is used for the final classification task.