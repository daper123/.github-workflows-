import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
import plotly.graph_objects as go
import plotly.express as px

# NEW: Import pybaseball
try:
    from pybaseball import pitching_stats, team_fielding
except ImportError:
    print("Pybaseball not installed. Please install it using: pip install pybaseball")
    # Fallback for environments where installation isn't possible
    pitching_stats = None
    team_fielding = None


# ==============================================================================
# Cell 1: Configuration & Setup
# ==============================================================================
class Config:
    """
    Centralized configuration settings for the entire system.
    This makes it easy to tune parameters and manage constants.
    """
    MLB_BASE_URL = "https://statsapi.mlb.com/api/v1"
    CURRENT_SEASON = datetime.now().year
    
    # Sabermetric thresholds
    MIN_PLATE_APPEARANCES = 50
    MIN_INNINGS_PITCHED = 20
    
    # Betting parameters
    KELLY_FRACTION = 0.25
    MIN_EDGE = 0.03
    MAX_BETS_PER_DAY = 5
    
    # Model parameters
    ROLLING_WINDOW = 14
    TRAIN_TEST_SPLIT = 0.2
    
    # Feature engineering settings
    PITCHER_FEATURES = ['SIERA', 'xFIP', 'K-BB%', 'WHIP', 'GB%']
    TEAM_FEATURES = ['wRC+', 'wOBA', 'ISO', 'K%', 'BB%']
    CONTEXT_FEATURES = ['park_factor', 'temperature', 'wind_speed', 'umpire_k_tendency']
    # NEW: Add Statcast features to the list
    STATCAST_PITCHER_FEATURES = ['xBA', 'xSLG', 'Barrel%']
    DEFENSE_FEATURES = ['DRS']


# Instantiate config for global use
config = Config()

# ==============================================================================
# Cell 2: Data Collection Module
# ==============================================================================
class MLBDataCollector:
    """Handles all data fetching from the MLB Stats API and Pybaseball."""
    def __init__(self):
        self.base_url = config.MLB_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'MLB-Prediction-Model/1.0'})
        # NEW: Cache for pybaseball data to avoid re-fetching during a single run
        self.pitcher_statcast_cache = None
        self.team_defense_cache = None

    def get_today_games(self):
        """Fetch today's MLB games schedule."""
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"{self.base_url}/schedule/games/?sportId=1&date={today}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            games = []
            for date in data.get('dates', []):
                for game in date.get('games', []):
                    if game['status']['abstractGameState'] == 'Preview':
                        games.append({
                            'game_id': game['gamePk'],
                            'home_team': game['teams']['home']['team']['name'],
                            'away_team': game['teams']['away']['team']['name'],
                            'home_team_id': game['teams']['home']['team']['id'],
                            'away_team_id': game['teams']['away']['team']['id'],
                            'game_time': game['gameDate'],
                            'venue_name': game['venue']['name'],
                            'venue_id': game['venue']['id']
                        })
            return pd.DataFrame(games)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching games: {e}")
            return pd.DataFrame()

    def get_probable_pitchers(self, game_id):
        """Get probable starting pitchers for a specific game."""
        url = f"{self.base_url}/game/{game_id}/boxscore"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            home_pitcher_info = data.get('teams', {}).get('home', {}).get('info', [])
            away_pitcher_info = data.get('teams', {}).get('away', {}).get('info', [])

            home_pitcher_name = next((p['fullName'] for p in home_pitcher_info if p.get('title') == 'Probable Pitcher'), None)
            away_pitcher_name = next((p['fullName'] for p in away_pitcher_info if p.get('title') == 'Probable Pitcher'), None)
            
            # This part is tricky as MLB API doesn't always provide IDs for probables easily.
            # We will rely on names for joining with pybaseball data.
            return {
                'home_pitcher_name': home_pitcher_name,
                'away_pitcher_name': away_pitcher_name
            }
        except requests.exceptions.RequestException as e:
            print(f"Error fetching pitchers for game {game_id}: {e}")
            return {}

    # NEW: Method to get pitcher Statcast data using pybaseball
    def get_pitcher_statcast_data(self, season=config.CURRENT_SEASON):
        """Fetches pitcher Statcast data from FanGraphs via pybaseball."""
        if self.pitcher_statcast_cache is not None:
            return self.pitcher_statcast_cache
        
        if pitching_stats is None:
            print("Cannot fetch Statcast data because pybaseball is not installed.")
            return pd.DataFrame()
            
        print(f"Fetching pitcher Statcast data for {season} season... (This may take a moment)")
        try:
            # Fetch data for pitchers with at least 20 innings pitched
            data = pitching_stats(season, qual=config.MIN_INNINGS_PITCHED)
            # Select relevant columns and clean up names
            data = data[['Name', 'Team', 'xBA', 'xSLG', 'Barrel%']]
            data['Barrel%'] = data['Barrel%'].str.rstrip('%').astype('float') / 100.0
            self.pitcher_statcast_cache = data
            print("Successfully fetched pitcher Statcast data.")
            return data
        except Exception as e:
            print(f"Error fetching pitcher Statcast data with pybaseball: {e}")
            return pd.DataFrame()

    # NEW: Method to get team defensive data using pybaseball
    def get_team_defense_data(self, season=config.CURRENT_SEASON):
        """Fetches team defensive stats (like DRS) from FanGraphs via pybaseball."""
        if self.team_defense_cache is not None:
            return self.team_defense_cache
            
        if team_fielding is None:
            print("Cannot fetch defense data because pybaseball is not installed.")
            return pd.DataFrame()

        print(f"Fetching team defensive data for {season} season...")
        try:
            data = team_fielding(season)
            # Select relevant columns
            data = data[['Team', 'DRS']]
            self.team_defense_cache = data
            print("Successfully fetched team defensive data.")
            return data
        except Exception as e:
            print(f"Error fetching team defense data with pybaseball: {e}")
            return pd.DataFrame()


    def get_pitcher_stats(self, pitcher_id, season=config.CURRENT_SEASON):
        """Fetch seasonal pitcher statistics."""
        url = f"{self.base_url}/people/{pitcher_id}/stats?stats=season&group=pitching&season={season}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('stats') and data['stats'][0].get('splits'):
                return data['stats'][0]['splits'][0].get('stat', {})
            return {}
        except requests.exceptions.RequestException as e:
            print(f"Error fetching pitcher stats for {pitcher_id}: {e}")
            return {}

    def get_team_stats(self, team_id, season=config.CURRENT_SEASON, group='hitting'):
        """Fetch seasonal team statistics (hitting or pitching)."""
        url = f"{self.base_url}/teams/{team_id}/stats?stats=season&group={group}&season={season}"
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get('stats') and data['stats'][0].get('splits'):
                return data['stats'][0]['splits'][0].get('stat', {})
            return {}
        except requests.exceptions.RequestException as e:
            print(f"Error fetching team stats for {team_id}: {e}")
            return {}

# ==============================================================================
# Cell 3: Sabermetric Calculator (No changes needed here)
# ==============================================================================
class SabermetricCalculator:
    """Contains methods to calculate advanced baseball statistics."""
    @staticmethod
    def _safe_get(stats, key, default=0):
        return float(stats.get(key, default))

    @classmethod
    def calculate_fip(cls, stats, fip_constant=3.2):
        hr = cls._safe_get(stats, 'homeRuns')
        bb = cls._safe_get(stats, 'baseOnBalls')
        hbp = cls._safe_get(stats, 'hitByPitch')
        k = cls._safe_get(stats, 'strikeOuts')
        ip = cls._safe_get(stats, 'inningsPitched', 1)
        if ip == 0: return None
        return ((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip + fip_constant

    @classmethod
    def calculate_xfip(cls, stats, league_hr_fb_rate=0.105, fip_constant=3.2):
        fb = cls._safe_get(stats, 'flyOuts')
        bb = cls._safe_get(stats, 'baseOnBalls')
        hbp = cls._safe_get(stats, 'hitByPitch')
        k = cls._safe_get(stats, 'strikeOuts')
        ip = cls._safe_get(stats, 'inningsPitched', 1)
        if ip == 0 or fb == 0: return None
        xHR = fb * league_hr_fb_rate
        return ((13 * xHR) + (3 * (bb + hbp)) - (2 * k)) / ip + fip_constant

    @classmethod
    def calculate_siera(cls, stats):
        k_per_pa = cls._safe_get(stats, 'strikeoutsPer9Inn') / 9
        bb_per_pa = cls._safe_get(stats, 'walksPer9Inn') / 9
        gb_rate = cls._safe_get(stats, 'groundoutsToAirouts')
        if (gb_rate + 1) == 0: return None
        gb_pct = gb_rate / (gb_rate + 1)
        siera = 5.50 - 10.5 * k_per_pa + 5.5 * bb_per_pa - 1.5 * gb_pct
        return round(siera, 3) if siera else None

    @classmethod
    def calculate_woba(cls, stats):
        weights = {'bb': 0.69, 'hbp': 0.72, 'single': 0.88, 'double': 1.24, 'triple': 1.56, 'hr': 2.00}
        pa = cls._safe_get(stats, 'plateAppearances', 1)
        if pa == 0: return None
        bb = cls._safe_get(stats, 'baseOnBalls')
        hbp = cls._safe_get(stats, 'hitByPitch')
        hits = cls._safe_get(stats, 'hits')
        doubles = cls._safe_get(stats, 'doubles')
        triples = cls._safe_get(stats, 'triples')
        hr = cls._safe_get(stats, 'homeRuns')
        singles = hits - doubles - triples - hr
        numerator = (weights['bb'] * bb + weights['hbp'] * hbp + weights['single'] * singles +
                     weights['double'] * doubles + weights['triple'] * triples + weights['hr'] * hr)
        return round(numerator / pa, 3)

    @staticmethod
    def calculate_wrc_plus(woba, league_woba=0.320, park_factor=1.0):
        if woba is None: return None
        wrc_plus = (((woba - league_woba) / 1.24) + 1) * 100 * park_factor
        return round(wrc_plus)

# ==============================================================================
# Cell 4: Contextual Data (No changes needed here)
# ==============================================================================
class ContextualDataCollector:
    @staticmethod
    def get_park_factors():
        return {
            'Coors Field': 1.15, 'Great American Ball Park': 1.10, 'Fenway Park': 1.05,
            'Globe Life Field': 1.04, 'Citizens Bank Park': 1.03, 'Yankee Stadium': 1.03,
            'T-Mobile Park': 0.94, 'Oracle Park': 0.92, 'Petco Park': 0.90,
        }

    @staticmethod
    def get_weather_data(venue_name):
        return {
            'temperature': np.random.uniform(60, 95),
            'wind_speed': np.random.uniform(0, 15),
            'wind_direction': np.random.choice(['In', 'Out', 'L-R', 'R-L']),
        }

    @staticmethod
    def get_umpire_tendencies(umpire_name):
        return {'umpire_k_impact': np.random.uniform(-0.02, 0.02)}

# ==============================================================================
# Cell 5: Feature Engineering
# ==============================================================================
class FeatureEngineer:
    """Creates the final feature set for the machine learning model."""
    def __init__(self, saber_calc, context_collector):
        self.saber = saber_calc
        self.context = context_collector

    # NEW: Updated method signature to accept new data
    def create_game_features(self, game_data, pitcher_stats, team_stats, pitcher_statcast, team_defense):
        """Create a comprehensive feature set for an F5 prediction."""
        features = {}

        # --- Pitcher Differentials (Traditional & Advanced) ---
        home_siera = self.saber.calculate_siera(pitcher_stats['home'])
        away_siera = self.saber.calculate_siera(pitcher_stats['away'])
        features['siera_diff'] = (away_siera - home_siera) if home_siera and away_siera else 0

        home_k_bb = (self.saber._safe_get(pitcher_stats['home'], 'strikeOuts') - 
                     self.saber._safe_get(pitcher_stats['home'], 'baseOnBalls'))
        away_k_bb = (self.saber._safe_get(pitcher_stats['away'], 'strikeOuts') - 
                     self.saber._safe_get(pitcher_stats['away'], 'baseOnBalls'))
        features['k_bb_diff'] = home_k_bb - away_k_bb
        
        # --- Offensive Differentials ---
        park_factor = self.context.get_park_factors().get(game_data.get('venue_name'), 1.0)
        home_woba = self.saber.calculate_woba(team_stats['home'])
        away_woba = self.saber.calculate_woba(team_stats['away'])
        home_wrc_plus = self.saber.calculate_wrc_plus(home_woba, park_factor=park_factor) or 100
        away_wrc_plus = self.saber.calculate_wrc_plus(away_woba, park_factor=park_factor) or 100
        features['wrc_plus_diff'] = home_wrc_plus - away_wrc_plus

        # --- NEW: Statcast Pitcher Feature Differentials ---
        home_pitcher_sc = pitcher_statcast[pitcher_statcast['Name'] == game_data['home_pitcher_name']]
        away_pitcher_sc = pitcher_statcast[pitcher_statcast['Name'] == game_data['away_pitcher_name']]

        if not home_pitcher_sc.empty and not away_pitcher_sc.empty:
            features['xba_diff'] = (away_pitcher_sc['xBA'].values[0] - home_pitcher_sc['xBA'].values[0])
            features['xslg_diff'] = (away_pitcher_sc['xSLG'].values[0] - home_pitcher_sc['xSLG'].values[0])
            features['barrel_diff'] = (away_pitcher_sc['Barrel%'].values[0] - home_pitcher_sc['Barrel%'].values[0])
        else:
            features['xba_diff'] = 0
            features['xslg_diff'] = 0
            features['barrel_diff'] = 0

        # --- NEW: Team Defense Feature Differential ---
        # Note: Need a mapping from full team name to the abbreviation used by pybaseball
        # This is a simplified example. A real implementation needs a robust mapping.
        team_name_map = { "New York Yankees": "Yankees", "Boston Red Sox": "Red Sox", "Los Angeles Dodgers": "Dodgers" } # etc.
        home_team_abbr = team_name_map.get(game_data['home_team'], game_data['home_team'])
        away_team_abbr = team_name_map.get(game_data['away_team'], game_data['away_team'])

        home_team_def = team_defense[team_defense['Team'] == home_team_abbr]
        away_team_def = team_defense[team_defense['Team'] == away_team_abbr]

        if not home_team_def.empty and not away_team_def.empty:
            features['drs_diff'] = (home_team_def['DRS'].values[0] - away_team_def['DRS'].values[0])
        else:
            features['drs_diff'] = 0


        # --- Contextual Features ---
        weather = self.context.get_weather_data(game_data['venue_name'])
        features['temperature'] = weather['temperature']
        features['wind_speed'] = weather['wind_speed']
        features['wind_out'] = 1 if weather['wind_direction'] == 'Out' else 0
        features['park_factor'] = park_factor
        
        return features

# ==============================================================================
# Cell 6: ML Models
# ==============================================================================
class F5PredictionModel:
    """Manages the training, evaluation, and prediction of ML models."""
    def __init__(self):
        self.xgb_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.feature_importance = None

    def create_demo_model(self):
        """Creates and trains a demo model on synthetic data."""
        print("Generating synthetic data for demo model...")
        
        # NEW: Update feature names to include new Statcast and defense features
        self.feature_names = [
            'siera_diff', 'k_bb_diff', 'wrc_plus_diff',
            'xba_diff', 'xslg_diff', 'barrel_diff', 'drs_diff',
            'temperature', 'wind_speed', 'wind_out', 'park_factor'
        ]
        
        num_samples = 500
        data = {feature: np.random.randn(num_samples) for feature in self.feature_names}
        
        y_signal = (data['siera_diff'] < 0) & (data['wrc_plus_diff'] > 0) & (data['barrel_diff'] < 0)
        y_noise = np.random.randint(0, 2, num_samples)
        y = np.where(np.random.rand(num_samples) < 0.2, y_noise, y_signal)

        demo_df = pd.DataFrame(data)
        
        print("Training demo models...")
        self.train_models(demo_df, pd.Series(y))

    def prepare_training_data(self, historical_games_df):
        """Prepares features (X) and labels (y) for training."""
        # This would be used if you had a historical CSV with all features pre-calculated
        self.feature_names = [col for col in historical_games_df.columns if '_diff' in col or col in config.CONTEXT_FEATURES]
        X = historical_games_df[self.feature_names]
        y = (historical_games_df['f5_home_score'] > historical_games_df['f5_away_score']).astype(int)
        return X, y

    def train_models(self, X, y):
        """Trains XGBoost and Random Forest models."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TRAIN_TEST_SPLIT, shuffle=False
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, objective='binary:logistic', random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.xgb_model.fit(X_train_scaled, y_train)

        self.rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.rf_model.fit(X_train_scaled, y_train)

        xgb_acc = accuracy_score(y_test, self.xgb_model.predict(X_test_scaled))
        rf_acc = accuracy_score(y_test, self.rf_model.predict(X_test_scaled))
        print(f"Demo Model Accuracy -> XGBoost: {xgb_acc:.3f}, Random Forest: {rf_acc:.3f}")

        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

    def predict_game(self, game_features_dict):
        """Generates F5 predictions for a single game."""
        if not self.xgb_model or not self.rf_model:
            print("Models are not trained. Cannot predict.")
            return None
        
        game_features_df = pd.DataFrame([game_features_dict], columns=self.feature_names)
        features_scaled = self.scaler.transform(game_features_df)

        xgb_prob = self.xgb_model.predict_proba(features_scaled)[0]
        rf_prob = self.rf_model.predict_proba(features_scaled)[0]
        ensemble_prob = (xgb_prob + rf_prob) / 2

        return {
            'home_win_prob': ensemble_prob[1],
            'away_win_prob': ensemble_prob[0],
        }

# ==============================================================================
# Cell 7: Betting Analysis (No changes needed here)
# ==============================================================================
class BettingAnalyzer:
    @staticmethod
    def calculate_implied_probability(odds):
        if odds > 0: return 100 / (odds + 100)
        return abs(odds) / (abs(odds) + 100)

    @staticmethod
    def kelly_criterion(prob, odds, fraction=config.KELLY_FRACTION):
        if odds > 0: decimal_odds = (odds / 100) + 1
        else: decimal_odds = (100 / abs(odds)) + 1
        b = decimal_odds - 1
        if b <= 0: return 0
        q = 1 - prob
        kelly_pct = (b * prob - q) / b
        return min(max(0, kelly_pct * fraction), 0.05)

    def find_opportunities(self, game_data, prediction, betting_odds):
        opportunities = []
        home_odds = betting_odds.get('home_f5_ml')
        if home_odds:
            implied_prob = self.calculate_implied_probability(home_odds)
            edge = prediction['home_win_prob'] - implied_prob
            if edge > config.MIN_EDGE:
                opportunities.append({
                    'game': f"{game_data['away_team']} @ {game_data['home_team']}",
                    'bet_on': game_data['home_team'], 'model_prob': prediction['home_win_prob'],
                    'implied_prob': implied_prob, 'edge': edge, 'odds': home_odds,
                    'kelly_size': self.kelly_criterion(prediction['home_win_prob'], home_odds)
                })
        away_odds = betting_odds.get('away_f5_ml')
        if away_odds:
            implied_prob = self.calculate_implied_probability(away_odds)
            edge = prediction['away_win_prob'] - implied_prob
            if edge > config.MIN_EDGE:
                opportunities.append({
                    'game': f"{game_data['away_team']} @ {game_data['home_team']}",
                    'bet_on': game_data['away_team'], 'model_prob': prediction['away_win_prob'],
                    'implied_prob': implied_prob, 'edge': edge, 'odds': away_odds,
                    'kelly_size': self.kelly_criterion(prediction['away_win_prob'], away_odds)
                })
        return opportunities

# ==============================================================================
# Cell 8: Visualization & Reporting (No changes needed here)
# ==============================================================================
class DashboardGenerator:
    @staticmethod
    def create_summary_report(opportunities_df):
        if opportunities_df.empty:
            print("\nNo valuable betting opportunities found today.")
            return
        print("\n" + "="*80)
        print("🎯 Top F5 Betting Opportunities for Today")
        print("="*80)
        report_df = opportunities_df.sort_values('edge', ascending=False).head(config.MAX_BETS_PER_DAY)
        for _, opp in report_df.iterrows():
            print(f"\nGame: {opp['game']}")
            print(f"  Bet On: {opp['bet_on']} F5 ML ({opp['odds']:+d})")
            print(f"  Model Edge: {opp['edge']:.2%}")
            print(f"  Model Prob: {opp['model_prob']:.2%} vs. Implied Prob: {opp['implied_prob']:.2%}")
            print(f"  Recommended Bet Size (Kelly): {opp['kelly_size']:.2%} of bankroll")

    @staticmethod
    def create_feature_importance_plot(model):
        if model.feature_importance is not None:
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=model.feature_importance.head(15), palette='viridis')
            plt.title('Top 15 Feature Importances (XGBoost)', fontsize=16)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.show()

# ==============================================================================
# Cell 9: Main Execution Pipeline
# ==============================================================================
class MLBBettingSystem:
    """Orchestrates the entire prediction and betting analysis pipeline."""
    def __init__(self):
        self.data_collector = MLBDataCollector()
        self.saber_calc = SabermetricCalculator()
        self.context = ContextualDataCollector()
        self.feature_eng = FeatureEngineer(self.saber_calc, self.context)
        self.model = F5PredictionModel()
        self.betting_analyzer = BettingAnalyzer()
        self.dashboard = DashboardGenerator()

    def run_daily_predictions(self, betting_odds_input):
        """Executes the complete daily pipeline."""
        print("🏁 Starting MLB F5 Prediction System...")
        
        if not self.model.xgb_model:
            print("\nℹ️ Models not trained. Creating and training a demo model...")
            self.model.create_demo_model()
            self.dashboard.create_feature_importance_plot(self.model)

        # NEW: Fetch advanced stats once per run
        pitcher_sc_data = self.data_collector.get_pitcher_statcast_data()
        team_def_data = self.data_collector.get_team_defense_data()
        if pitcher_sc_data.empty or team_def_data.empty:
            print("\n⚠️ Could not fetch advanced data. Predictions will be less accurate.")
            # Allow fallback to basic features if needed, or exit
            return

        print("\n📅 Fetching today's games...")
        games_df = self.data_collector.get_today_games()
        if games_df.empty:
            print("No games found for today. Exiting.")
            return
        print(f"Found {len(games_df)} games scheduled.")

        all_opportunities = []
        for _, game in games_df.iterrows():
            print(f"\n🎮 Processing: {game['away_team']} @ {game['home_team']}")
            
            pitchers = self.data_collector.get_probable_pitchers(game['game_id'])
            if not pitchers.get('home_pitcher_name') or not pitchers.get('away_pitcher_name'):
                print("  ⚠️ Pitchers not announced. Skipping game.")
                continue
            
            # Add pitcher names to game data for joining
            game['home_pitcher_name'] = pitchers['home_pitcher_name']
            game['away_pitcher_name'] = pitchers['away_pitcher_name']

            # Use placeholder stats as the MLB API part is complex
            pitcher_stats = {'home': {}, 'away': {}}
            team_stats = {'home': {}, 'away': {}}

            # NEW: Pass advanced data to the feature engineer
            features = self.feature_eng.create_game_features(game, pitcher_stats, team_stats, pitcher_sc_data, team_def_data)
            prediction = self.model.predict_game(features)
            
            if prediction and str(game['game_id']) in betting_odds_input:
                odds = betting_odds_input[str(game['game_id'])]
                opportunities = self.betting_analyzer.find_opportunities(game, prediction, odds)
                if opportunities:
                    all_opportunities.extend(opportunities)

        self.dashboard.create_summary_report(pd.DataFrame(all_opportunities))
        print("\n✅ Analysis Complete!")

# ==============================================================================
# Cell 10: Example Usage
# ==============================================================================
if __name__ == '__main__':
    # Ensure you have pybaseball installed: pip install pybaseball
    system = MLBBettingSystem()

    sample_odds = {
        '745281': { 'home_f5_ml': -130, 'away_f5_ml': +110 },
        '745280': { 'home_f5_ml': +140, 'away_f5_ml': -160 }
    }

    system.run_daily_predictions(betting_odds_input=sample_odds)