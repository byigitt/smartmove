import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class MetroPassengerPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def prepare_features(self, df):
        """Prepare features for the model with enhanced engineering"""
        # Convert timestamp to datetime if it's not already
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        # Extract time-based features
        df['Hour'] = df['Timestamp'].dt.hour
        df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
        df['Month'] = df['Timestamp'].dt.month
        df['DayOfMonth'] = df['Timestamp'].dt.day
        
        # Create cyclical time features
        df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour']/24)
        df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour']/24)
        df['DayOfWeek_Sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_Cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        
        # Extract station number from Station_ID
        df['Station_Num'] = df['Station_ID'].str.extract(r'(\d+)').astype(int)
        
        # Calculate distance from central station (S29)
        df['Distance_From_Central'] = abs(df['Station_Num'] - 29)
        
        return df
    
    def create_feature_pipeline(self):
        """Create an enhanced feature preprocessing pipeline"""
        # Define feature groups
        numeric_features = [
            'Hour', 'Hour_Sin', 'Hour_Cos', 
            'DayOfWeek_Sin', 'DayOfWeek_Cos',
            'Station_Num', 'Distance_From_Central',
            'Boarding_Passengers', 'Alighting_Passengers',
            'Transfer_Out', 'Capacity_Utilization'
        ]
        
        categorical_features = [
            'Metro_Line', 'Station_Type', 'Time_Period'
        ]
        
        boolean_features = ['Is_Weekend']
        
        # Create preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features),
                ('bool', 'passthrough', boolean_features)
            ],
            remainder='drop'  # Drop any other columns not specified
        )
        
        return preprocessor
    
    def build_model(self, model_type='rf'):
        """Build the model pipeline with specified type"""
        preprocessor = self.create_feature_pipeline()
        
        if model_type == 'rf':
            regressor = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )
        else:  # gradient boosting
            regressor = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            )
        
        return Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', regressor)
        ])
    
    def train(self, df, model_type='rf'):
        """Train the model with cross-validation"""
        print("Preparing features...")
        df = self.prepare_features(df)
        
        # Define features and target
        feature_columns = [
            'Hour', 'Hour_Sin', 'Hour_Cos', 
            'DayOfWeek_Sin', 'DayOfWeek_Cos',
            'Station_Num', 'Distance_From_Central',
            'Metro_Line', 'Station_Type', 'Time_Period',
            'Is_Weekend', 'Boarding_Passengers',
            'Alighting_Passengers', 'Transfer_Out',
            'Capacity_Utilization'
        ]
        
        target_column = 'Current_Load'
        
        # Split features and target
        X = df[feature_columns]
        y = df[target_column]
        
        # Build and train model
        print(f"Training {model_type.upper()} model...")
        self.model = self.build_model(model_type)
        
        # Perform time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(
            self.model, X, y, 
            cv=tscv, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        print("\nCross-validation RMSE scores:")
        print(f"Mean: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Final train-test split for detailed evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Keep temporal order
        )
        
        # Train final model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        self._evaluate_model(X_test, y_test)
        
        # Store feature names for later use
        self.feature_names = (
            feature_columns +
            self.model.named_steps['preprocessor']
            .named_transformers_['cat']
            .get_feature_names_out(categorical_features).tolist()
        )
        
        return self.model
    
    def _evaluate_model(self, X_test, y_test):
        """Perform detailed model evaluation"""
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        print("\nModel Performance Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.4%}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Passenger Load')
        plt.ylabel('Predicted Passenger Load')
        plt.title('Actual vs Predicted Passenger Load')
        plt.tight_layout()
        plt.savefig('prediction_performance.png')
        plt.close()
        
        # Feature importance analysis
        if hasattr(self.model['regressor'], 'feature_importances_'):
            self._plot_feature_importance()
    
    def _plot_feature_importance(self):
        """Plot feature importance"""
        importances = self.model['regressor'].feature_importances_
        indices = np.argsort(importances)[-15:]  # Top 15 features
        
        plt.figure(figsize=(10, 6))
        plt.title('Top 15 Feature Importances')
        plt.barh(range(15), importances[indices])
        plt.yticks(range(15), [self.feature_names[i] for i in indices])
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()
    
    def predict_passenger_load(self, features_df):
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Prepare features
        features_df = self.prepare_features(features_df)
        
        return self.model.predict(features_df)
    
    def save_model(self, filename='metro_passenger_model.joblib'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        joblib.dump(self.model, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='metro_passenger_model.joblib'):
        """Load a trained model"""
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")

def main():
    # Load the data
    print("Loading data...")
    df = pd.read_csv('metro_passenger_data.csv')
    
    # Initialize and train the model
    predictor = MetroPassengerPredictor()
    predictor.train(df, model_type='rf')
    
    # Save the model
    predictor.save_model()
    
    # Make a sample prediction
    print("\nMaking sample predictions...")
    sample_data = pd.DataFrame({
        'Timestamp': [datetime.now()],
        'Metro_Line': ['M2'],
        'Station_ID': ['S29'],
        'Station_Type': ['central'],
        'Boarding_Passengers': [100],
        'Alighting_Passengers': [50],
        'Transfer_Out': [20],
        'Capacity_Utilization': [60.0],
        'Is_Weekend': [0],
        'Time_Period': ['peak']
    })
    
    prediction = predictor.predict_passenger_load(sample_data)
    print(f"Predicted passenger load: {prediction[0]:.0f}")

if __name__ == "__main__":
    main() 