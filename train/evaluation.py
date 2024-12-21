"""
Model evaluation functions for the Ankara Metro prediction system
"""

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """
    Perform detailed model evaluation
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE only for non-zero actual values to avoid division by zero
    mask = y_test != 0
    mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask])
    
    metrics = {
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    print("\nModel Performance Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"MAPE: {mape:.2%}")
    
    # Plot actual vs predicted with density
    plt.figure(figsize=(12, 8))
    plt.hexbin(y_test, y_pred, gridsize=30, cmap='YlOrRd')
    plt.colorbar(label='Count')
    plt.plot([y_test.min(), y_test.max()], 
            [y_test.min(), y_test.max()], 
            'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Occupancy Rate')
    plt.ylabel('Predicted Occupancy Rate')
    plt.title('Actual vs Predicted Occupancy Rate\nwith Density Visualization')
    plt.legend()
    plt.tight_layout()
    plt.savefig('prediction_performance.png')
    plt.close()
    
    return metrics

def interpret_occupancy(prediction):
    """
    Provide interpretation of occupancy rate prediction
    
    Args:
        prediction (float): Predicted occupancy rate
        
    Returns:
        str: Status description
    """
    if prediction < 30:
        return "Low occupancy - Very comfortable"
    elif prediction < 50:
        return "Moderate occupancy - Comfortable"
    elif prediction < 70:
        return "High occupancy - Somewhat crowded"
    elif prediction < 85:
        return "Very high occupancy - Crowded"
    else:
        return "Extremely high occupancy - Very crowded" 