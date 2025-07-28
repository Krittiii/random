#!/usr/bin/env python3
"""
Complete example script for GPS satellite parameter prediction with SBAS refinement
Replace the data loading section with your actual 1-week GPS dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from gps_prediction_system import GPSPredictionSystem
from data_loader import GPSDataLoader, load_your_data

def load_actual_data(filepath):
    """
    Load your actual 1-week GPS dataset
    
    Parameters:
    -----------
    filepath : str
        Path to your GPS data file (CSV, Excel, etc.)
    
    Returns:
    --------
    pd.DataFrame
        Processed GPS data
    """
    print("Loading your GPS dataset...")
    
    try:
        # Use the data loader to load and preprocess your data
        data, stats = load_your_data(filepath)
        
        print(f"Successfully loaded {len(data)} records")
        print(f"Data covers {stats['unique_prns']} unique satellites")
        print(f"Week range: {stats['week_range'][0]} to {stats['week_range'][1]}")
        
        return data
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        print("Using sample data for demonstration...")
        
        # Fallback to sample data if actual data loading fails
        from gps_prediction_system import create_sample_data
        return create_sample_data(n_samples=5000)

def evaluate_model_performance(predictions, actual_data):
    """
    Comprehensive model evaluation
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    print("\n=== MODEL PERFORMANCE EVALUATION ===")
    
    # Evaluate S4 predictions
    if 'TOTAL_S4_L1' in actual_data.columns:
        y_true_s4 = actual_data['TOTAL_S4_L1'].values
        y_pred_s4_initial = predictions['S4_initial'].values
        y_pred_s4_refined = predictions['S4_refined'].values
        y_pred_s4_final = predictions['S4_final'].values
        
        print("\nS4 Prediction Performance:")
        print("-" * 30)
        
        # Initial S4 metrics
        s4_initial_rmse = np.sqrt(mean_squared_error(y_true_s4, y_pred_s4_initial))
        s4_initial_mae = mean_absolute_error(y_true_s4, y_pred_s4_initial)
        s4_initial_r2 = r2_score(y_true_s4, y_pred_s4_initial)
        
        print(f"Initial S4 Prediction:")
        print(f"  RMSE: {s4_initial_rmse:.6f}")
        print(f"  MAE:  {s4_initial_mae:.6f}")
        print(f"  R²:   {s4_initial_r2:.6f}")
        
        # SBAS refined S4 metrics
        s4_refined_rmse = np.sqrt(mean_squared_error(y_true_s4, y_pred_s4_refined))
        s4_refined_mae = mean_absolute_error(y_true_s4, y_pred_s4_refined)
        s4_refined_r2 = r2_score(y_true_s4, y_pred_s4_refined)
        
        print(f"\nSBAS Refined S4:")
        print(f"  RMSE: {s4_refined_rmse:.6f}")
        print(f"  MAE:  {s4_refined_mae:.6f}")
        print(f"  R²:   {s4_refined_r2:.6f}")
        
        # Final S4 metrics
        s4_final_rmse = np.sqrt(mean_squared_error(y_true_s4, y_pred_s4_final))
        s4_final_mae = mean_absolute_error(y_true_s4, y_pred_s4_final)
        s4_final_r2 = r2_score(y_true_s4, y_pred_s4_final)
        
        print(f"\nFinal S4 Re-estimation:")
        print(f"  RMSE: {s4_final_rmse:.6f}")
        print(f"  MAE:  {s4_final_mae:.6f}")
        print(f"  R²:   {s4_final_r2:.6f}")
        
        # Improvement analysis
        rmse_improvement = ((s4_initial_rmse - s4_final_rmse) / s4_initial_rmse) * 100
        print(f"\nRMSE Improvement: {rmse_improvement:.2f}%")
    
    # Evaluate other parameters
    for param, pred_col in [('60s_sigma', 'Sigma_predicted'), ('CNO', 'CNO_predicted')]:
        if param in actual_data.columns:
            y_true = actual_data[param].values
            y_pred = predictions[pred_col].values
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            print(f"\n{param} Prediction:")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE:  {mae:.6f}")
            print(f"  R²:   {r2:.6f}")

def create_detailed_plots(predictions, actual_data=None):
    """
    Create comprehensive visualization of results
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: S4 Evolution Through Pipeline
    plt.subplot(3, 3, 1)
    plt.plot(predictions['S4_initial'], label='Initial S4', alpha=0.7, linewidth=1)
    plt.plot(predictions['S4_refined'], label='SBAS Refined S4', alpha=0.7, linewidth=1)
    plt.plot(predictions['S4_final'], label='Final S4', alpha=0.7, linewidth=1)
    if actual_data is not None and 'TOTAL_S4_L1' in actual_data.columns:
        plt.plot(actual_data['TOTAL_S4_L1'], label='Actual S4', alpha=0.8, linewidth=1)
    plt.title('S4 Prediction Evolution')
    plt.ylabel('S4 Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: S4 Scatter Plot (Final vs Actual)
    plt.subplot(3, 3, 2)
    if actual_data is not None and 'TOTAL_S4_L1' in actual_data.columns:
        plt.scatter(actual_data['TOTAL_S4_L1'], predictions['S4_final'], 
                   alpha=0.6, s=20, c=predictions['ELV'], cmap='viridis')
        plt.plot([actual_data['TOTAL_S4_L1'].min(), actual_data['TOTAL_S4_L1'].max()],
                [actual_data['TOTAL_S4_L1'].min(), actual_data['TOTAL_S4_L1'].max()], 
                'r--', linewidth=2)
        plt.xlabel('Actual S4')
        plt.ylabel('Predicted S4')
        plt.title('S4: Predicted vs Actual')
        plt.colorbar(label='Elevation (deg)')
    else:
        plt.hist(predictions['S4_final'], bins=50, alpha=0.7)
        plt.title('Final S4 Distribution')
        plt.xlabel('S4 Value')
        plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Sigma Predictions
    plt.subplot(3, 3, 3)
    plt.plot(predictions['Sigma_predicted'], label='Predicted Sigma', color='orange', linewidth=1)
    if actual_data is not None and '60s_sigma' in actual_data.columns:
        plt.plot(actual_data['60s_sigma'], label='Actual Sigma', alpha=0.7, linewidth=1)
    plt.title('60s Sigma Predictions')
    plt.ylabel('Sigma Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: CNO Predictions
    plt.subplot(3, 3, 4)
    plt.plot(predictions['CNO_predicted'], label='Predicted CNO', color='green', linewidth=1)
    if actual_data is not None and 'CNO' in actual_data.columns:
        plt.plot(actual_data['CNO'], label='Actual CNO', alpha=0.7, linewidth=1)
    plt.title('CNO Predictions')
    plt.ylabel('CNO (dB-Hz)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: SBAS Distance vs S4 Final
    plt.subplot(3, 3, 5)
    scatter = plt.scatter(predictions['SBAS_distance'], predictions['S4_final'], 
                         c=predictions['PRN_NO'], cmap='tab10', alpha=0.6, s=30)
    plt.xlabel('SBAS Distance')
    plt.ylabel('Final S4')
    plt.title('S4 Final vs SBAS Distance')
    plt.colorbar(scatter, label='PRN Number')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Elevation vs S4 Relationship
    plt.subplot(3, 3, 6)
    plt.scatter(predictions['ELV'], predictions['S4_final'], 
               c=predictions['AZM'], cmap='hsv', alpha=0.6, s=20)
    plt.xlabel('Elevation (degrees)')
    plt.ylabel('Final S4')
    plt.title('S4 vs Elevation')
    plt.colorbar(label='Azimuth (deg)')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: PRN-wise S4 Distribution
    plt.subplot(3, 3, 7)
    unique_prns = sorted(predictions['PRN_NO'].unique())
    s4_by_prn = [predictions[predictions['PRN_NO'] == prn]['S4_final'].values 
                 for prn in unique_prns]
    plt.boxplot(s4_by_prn, labels=unique_prns)
    plt.xlabel('PRN Number')
    plt.ylabel('Final S4')
    plt.title('S4 Distribution by Satellite')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 8: SBAS Satellite Usage
    plt.subplot(3, 3, 8)
    sbas_counts = predictions['nearest_SBAS'].value_counts()
    plt.pie(sbas_counts.values, labels=sbas_counts.index, autopct='%1.1f%%')
    plt.title('SBAS Satellite Usage')
    
    # Plot 9: Residuals Analysis
    plt.subplot(3, 3, 9)
    if actual_data is not None and 'TOTAL_S4_L1' in actual_data.columns:
        # Reset index to ensure alignment
        actual_reset = actual_data.reset_index(drop=True)
        predictions_reset = predictions.reset_index(drop=True)
        residuals = actual_reset['TOTAL_S4_L1'] - predictions_reset['S4_final']
        plt.scatter(predictions_reset['S4_final'], residuals, alpha=0.6, s=20)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted S4')
        plt.ylabel('Residuals')
        plt.title('Residuals Analysis')
        plt.grid(True, alpha=0.3)
    else:
        plt.hist(predictions['S4_final'], bins=30, alpha=0.7)
        plt.title('S4 Final Distribution')
        plt.xlabel('S4 Value')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('gps_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_detailed_results(predictions, actual_data, filepath='detailed_results.csv'):
    """
    Save detailed results including all intermediate predictions
    """
    # Combine predictions with actual data
    if actual_data is not None:
        results = predictions.copy()
        
        # Add actual values for comparison
        for col in ['TOTAL_S4_L1', '60s_sigma', 'CNO']:
            if col in actual_data.columns:
                results[f'Actual_{col}'] = actual_data[col].values
        
        # Calculate errors
        if 'TOTAL_S4_L1' in actual_data.columns:
            results['S4_Initial_Error'] = actual_data['TOTAL_S4_L1'] - results['S4_initial']
            results['S4_Refined_Error'] = actual_data['TOTAL_S4_L1'] - results['S4_refined']
            results['S4_Final_Error'] = actual_data['TOTAL_S4_L1'] - results['S4_final']
        
        if '60s_sigma' in actual_data.columns:
            results['Sigma_Error'] = actual_data['60s_sigma'] - results['Sigma_predicted']
        
        if 'CNO' in actual_data.columns:
            results['CNO_Error'] = actual_data['CNO'] - results['CNO_predicted']
    else:
        results = predictions
    
    # Save to CSV
    results.to_csv(filepath, index=False)
    print(f"\nDetailed results saved to: {filepath}")
    
    return results

def main():
    """
    Main execution function
    """
    print("GPS Satellite Parameter Prediction System with SBAS Refinement")
    print("=" * 70)
    
    # Step 1: Load your GPS data
    # REPLACE THIS PATH WITH YOUR ACTUAL DATA FILE
    data_filepath = "your_gps_week_data.csv"  # Change this to your file path
    
    try:
        # Load actual data
        data = load_actual_data(data_filepath)
    except:
        print("Could not load actual data. Creating sample data for demonstration...")
        from gps_prediction_system import create_sample_data
        data = create_sample_data(n_samples=3000)
    
    print(f"\nDataset Overview:")
    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    # Step 2: Split data into training and testing
    print("\nSplitting data into train/test sets...")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, 
                                           stratify=data['PRN_NO'])
    
    print(f"Training data: {train_data.shape[0]} samples")
    print(f"Testing data: {test_data.shape[0]} samples")
    
    # Step 3: Initialize and train the prediction system
    print("\nInitializing GPS Prediction System...")
    gps_system = GPSPredictionSystem()
    
    # Train the system
    gps_system.train(train_data)
    
    # Step 4: Make predictions on test data
    print("\nMaking predictions on test data...")
    predictions = gps_system.predict(test_data)
    
    # Step 5: Evaluate performance
    evaluate_model_performance(predictions, test_data)
    
    # Step 6: Create visualizations
    print("\nGenerating visualizations...")
    create_detailed_plots(predictions, test_data)
    
    # Step 7: Save results
    print("\nSaving results...")
    detailed_results = save_detailed_results(predictions, test_data)
    
    # Step 8: Save the trained model
    gps_system.save_model('trained_gps_model.pkl')
    
    # Step 9: Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Processed {len(data)} GPS observations")
    print(f"✓ Trained models on {len(train_data)} samples")
    print(f"✓ Generated predictions for {len(test_data)} samples")
    print(f"✓ Used {len(predictions['nearest_SBAS'].unique())} SBAS satellites for refinement")
    print(f"✓ Results saved to: detailed_results.csv")
    print(f"✓ Visualizations saved to: gps_prediction_results.png")
    print(f"✓ Trained model saved to: trained_gps_model.pkl")
    
    # Step 10: Show sample predictions
    print("\nSample Predictions:")
    print("-" * 50)
    sample_results = predictions.head(10)[['PRN_NO', 'AZM', 'ELV', 'S4_initial', 
                                          'S4_refined', 'S4_final', 'nearest_SBAS']]
    print(sample_results.to_string(index=False))
    
    return gps_system, predictions, detailed_results

def predict_new_data(model_path, new_data_path):
    """
    Load a trained model and make predictions on new data
    """
    print("Loading trained model and making predictions on new data...")
    
    # Load the trained model
    gps_system = GPSPredictionSystem()
    gps_system.load_model(model_path)
    
    # Load new data
    new_data = load_actual_data(new_data_path)
    
    # Make predictions
    predictions = gps_system.predict(new_data)
    
    # Save predictions
    predictions.to_csv('new_predictions.csv', index=False)
    print("New predictions saved to: new_predictions.csv")
    
    return predictions

if __name__ == "__main__":
    # Run the main prediction pipeline
    system, results, detailed = main()
    
    # Example of using the trained model on new data
    # predictions = predict_new_data('trained_gps_model.pkl', 'new_gps_data.csv')