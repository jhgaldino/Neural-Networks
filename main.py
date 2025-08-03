import pandas as pd
import numpy as np
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_and_preprocess_data(csv_file):
    """Load and preprocess football match data"""
    # Load data
    df = pd.read_csv(csv_file)
    
    # Drop unnecessary columns
    df = df.drop(['Div', 'Date', 'HTR', 'HTHG', 'HTAG', 'Referee'], axis=1, errors='ignore')
    
    # Remove last 39 columns (betting odds, etc.)
    df = df.iloc[:, :-39]
    
    # Split into training and test sets (last 10 matches for testing)
    training = df.iloc[:-10]
    test = df.iloc[-10:]
    
    # Shuffle training data
    training = training.sample(frac=1, random_state=42)
    test = test.sample(frac=1, random_state=42)
    
    return df, training, test

def prepare_features_and_targets(df):
    """Prepare features and targets for neural network"""
    # Separate team names
    team_names = df[['HomeTeam', 'AwayTeam']]
    
    # Get match statistics (excluding team names and goals)
    stats_df = df.iloc[:, 2:]
    
    # Convert match results to numerical values
    result_mapping = {"H": 1, "D": 0, "A": -1}
    stats_df_converted = stats_df.replace(result_mapping).infer_objects(copy=False)
    
    # Separate features and target
    # Remove FTHG, FTAG (actual goals) - we only use match statistics
    features = stats_df_converted.iloc[:, 2:]  # Skip FTHG, FTAG, start from FTR+1
    target = stats_df_converted[['FTR']]
    
    return features, target, team_names

def normalize_features(features):
    """Normalize features to 0-1 range"""
    features_normalized = features.copy()
    
    for col in features_normalized.columns:
        max_val = features_normalized[col].max()
        if max_val < 10:
            features_normalized[col] = features_normalized[col] / 10
        elif max_val < 100:
            features_normalized[col] = features_normalized[col] / 100
        else:
            print(f"Warning: Column {col} has max value {max_val}")
    
    return features_normalized

class FootballPredictor:
    """Simple neural network for football match prediction using scikit-learn"""
    
    def __init__(self, hidden_layer_sizes=(100,), learning_rate=0.01, max_iter=500):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=learning_rate,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, X, y):
        """Train the neural network"""
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y.ravel())
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Convert continuous predictions to discrete classes
        # Round to nearest integer and clip to valid range [-1, 1]
        predictions_discrete = np.round(predictions).astype(int)
        predictions_discrete = np.clip(predictions_discrete, -1, 1)
        
        return predictions_discrete
    
    def predict_proba(self, X):
        """Get raw prediction scores"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    raw_predictions = model.predict_proba(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, raw_predictions)
    
    # Accuracy for exact match prediction
    accuracy = accuracy_score(y_test, predictions)
    
    return {
        'mse': mse,
        'accuracy': accuracy,
        'predictions': predictions,
        'raw_predictions': raw_predictions
    }

def plot_results(y_true, y_pred, team_names, errors=None):
    """Plot training results and predictions"""
    plt.figure(figsize=(15, 5))
    
    # Plot training errors if available
    if errors is not None:
        plt.subplot(1, 2, 1)
        plt.plot(errors)
        plt.title('Training Loss')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
    
    # Plot predictions vs actual
    subplot_idx = 2 if errors is not None else 1
    plt.subplot(1, subplot_idx, subplot_idx)
    
    x_pos = range(len(y_true))
    plt.plot(x_pos, y_true, 'yo', label='Actual', markersize=10)
    plt.plot(x_pos, y_pred, 'b+', label='Predicted', markersize=10)
    
    # Add team names
    for i, (home, away) in enumerate(zip(team_names['HomeTeam'], team_names['AwayTeam'])):
        plt.text(i, 1.3, home[:8], rotation=45, ha='center')
        plt.text(i, -1.3, away[:8], rotation=-45, ha='center', va='top')
    
    plt.ylim(-2, 2)
    plt.ylabel('Match Result')
    plt.xlabel('Match')
    plt.title('Predictions vs Actual Results')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def hyperparameter_search(X_train, y_train):
    """Perform hyperparameter optimization"""
    param_dist = {
        'hidden_layer_sizes': [(50,), (100,), (150,), (100, 50), (150, 100)],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [300, 500, 1000]
    }
    
    mlp = MLPRegressor(random_state=42, early_stopping=True)
    
    # Use RandomizedSearchCV for hyperparameter optimization
    search = RandomizedSearchCV(
        mlp, 
        param_dist, 
        n_iter=10, 
        cv=3, 
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_train, y_train.ravel())
    
    return search.best_estimator_, search.best_params_

# Example usage
def main():
    """Main execution function"""
    # Note: You'll need to provide the path to your CSV file
    csv_file = 'E0.csv'  # Update this path
    
    try:
        # Load and preprocess data
        df, training_data, test_data = load_and_preprocess_data(csv_file)
        print(f"Data loaded successfully. Training: {len(training_data)}, Test: {len(test_data)}")
        
        # Prepare features and targets
        features, target, team_names = prepare_features_and_targets(df)
        
        # Normalize features
        features_normalized = normalize_features(features)
        
        # Split into training and test sets
        X_train = features_normalized.iloc[:-10]
        y_train = target.iloc[:-10]
        X_test = features_normalized.iloc[-10:]
        y_test = target.iloc[-10:]
        team_names_test = team_names.iloc[-10:]
        
        print(f"Feature shape: {X_train.shape}")
        print(f"Target shape: {y_train.shape}")
        
        # Train model
        model = FootballPredictor(hidden_layer_sizes=(100, 50), max_iter=500)
        model.fit(X_train, y_train.values.ravel())
        
        # Evaluate model
        results = evaluate_model(model, X_test, y_test)
        
        print(f"\nModel Performance:")
        print(f"MSE: {results['mse']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        
        # Display predictions
        print("\nPredictions:")
        for i, (_, row) in enumerate(team_names_test.iterrows()):
            actual = y_test.iloc[i, 0]
            predicted = results['predictions'][i]
            confidence = results['raw_predictions'][i]
            
            result_map = {1: 'Home Win', 0: 'Draw', -1: 'Away Win'}
            print(f"{row['HomeTeam']} vs {row['AwayTeam']}: "
                  f"Actual: {result_map[actual]}, "
                  f"Predicted: {result_map[predicted]} "
                  f"(confidence: {confidence:.2f})")
        
        # Plot results
        plot_results(y_test.values.ravel(), results['predictions'], team_names_test)
        
        # Optional: Hyperparameter optimization
        print("\nPerforming hyperparameter optimization...")
        best_model, best_params = hyperparameter_search(X_train, y_train.values.ravel())
        print(f"Best parameters: {best_params}")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        print("Please make sure the CSV file exists and update the file path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()