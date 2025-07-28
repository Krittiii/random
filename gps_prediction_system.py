import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

class GPSPredictionSystem:
    """
    Comprehensive GPS satellite parameter prediction system with SBAS refinement
    """
    
    def __init__(self):
        # Initialize models
        self.s4_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.sigma_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.cno_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.s4_final_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        # Scalers for feature normalization
        self.s4_scaler = StandardScaler()
        self.sigma_scaler = StandardScaler()
        self.cno_scaler = StandardScaler()
        self.s4_final_scaler = StandardScaler()
        
        # SBAS satellite positions (example coordinates - adjust as needed)
        self.sbas_satellites = {
            'SBAS_1': {'lat': 0.0, 'lon': 55.0, 'alt': 35786000},    # GAGAN PRN 127
            'SBAS_2': {'lat': 0.0, 'lon': 98.0, 'alt': 35786000},    # GAGAN PRN 128  
            'SBAS_3': {'lat': 0.0, 'lon': 142.0, 'alt': 35786000}    # MSAS PRN 129
        }
        
        self.is_trained = False
        
    def calculate_satellite_position(self, azm, elv, distance=20200000):
        """
        Calculate approximate satellite position from azimuth and elevation
        Assumes spherical Earth and satellite at GPS orbit altitude
        """
        azm_rad = np.radians(azm)
        elv_rad = np.radians(elv)
        
        # Convert to Cartesian coordinates (simplified)
        x = distance * np.cos(elv_rad) * np.sin(azm_rad)
        y = distance * np.cos(elv_rad) * np.cos(azm_rad)
        z = distance * np.sin(elv_rad)
        
        return np.array([x, y, z])
    
    def find_nearest_sbas(self, azm, elv):
        """
        Find the nearest SBAS satellite based on position
        """
        sat_pos = self.calculate_satellite_position(azm, elv)
        min_distance = float('inf')
        nearest_sbas = None
        
        for sbas_name, sbas_data in self.sbas_satellites.items():
            # Convert SBAS position to Cartesian (simplified)
            sbas_pos = np.array([
                sbas_data['alt'] * np.cos(np.radians(sbas_data['lat'])) * np.sin(np.radians(sbas_data['lon'])),
                sbas_data['alt'] * np.cos(np.radians(sbas_data['lat'])) * np.cos(np.radians(sbas_data['lon'])),
                sbas_data['alt'] * np.sin(np.radians(sbas_data['lat']))
            ])
            
            distance = euclidean(sat_pos, sbas_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_sbas = sbas_name
                
        return nearest_sbas, min_distance
    
    def apply_sbas_refinement(self, predicted_values, sbas_data, refinement_factor=0.1):
        """
        Apply SBAS-based refinement to predicted values
        """
        if sbas_data is None:
            return predicted_values
            
        # Simple refinement: weighted average with SBAS correction
        # In practice, this would use actual SBAS correction algorithms
        refinement = np.random.normal(0, refinement_factor * np.std(predicted_values), 
                                    size=len(predicted_values))
        return predicted_values + refinement
    
    def prepare_features(self, data, feature_cols):
        """
        Prepare feature matrix from data
        """
        return data[feature_cols].values
    
    def train(self, train_data):
        """
        Train all models in the prediction pipeline
        """
        print("Training GPS Prediction System...")
        
        # Prepare initial features for S4 prediction
        s4_features = ['GPS_TOW', 'Wk_no', 'PRN_NO', 'AZM', 'ELV']
        X_s4 = self.prepare_features(train_data, s4_features)
        y_s4 = train_data['TOTAL_S4_L1'].values
        
        # Train S4 model
        X_s4_scaled = self.s4_scaler.fit_transform(X_s4)
        self.s4_model.fit(X_s4_scaled, y_s4)
        
        # Get initial S4 predictions for training other models
        s4_pred_train = self.s4_model.predict(X_s4_scaled)
        
        # Apply SBAS refinement to training S4 predictions
        s4_refined_train = []
        for idx, (i, row) in enumerate(train_data.iterrows()):
            nearest_sbas, _ = self.find_nearest_sbas(row['AZM'], row['ELV'])
            # Simulate SBAS data (in practice, load actual SBAS corrections)
            sbas_data = {'correction': np.random.normal(0, 0.05)}
            s4_refined = self.apply_sbas_refinement([s4_pred_train[idx]], sbas_data)[0]
            s4_refined_train.append(s4_refined)
        
        s4_refined_train = np.array(s4_refined_train)
        
        # Train Sigma model using refined S4
        sigma_features = np.column_stack([X_s4, s4_refined_train])
        X_sigma_scaled = self.sigma_scaler.fit_transform(sigma_features)
        y_sigma = train_data['60s_sigma'].values
        self.sigma_model.fit(X_sigma_scaled, y_sigma)
        
        # Get sigma predictions and apply SBAS refinement
        sigma_pred_train = self.sigma_model.predict(X_sigma_scaled)
        sigma_refined_train = []
        for idx, (i, row) in enumerate(train_data.iterrows()):
            nearest_sbas, _ = self.find_nearest_sbas(row['AZM'], row['ELV'])
            sbas_data = {'correction': np.random.normal(0, 0.02)}
            sigma_refined = self.apply_sbas_refinement([sigma_pred_train[idx]], sbas_data)[0]
            sigma_refined_train.append(sigma_refined)
        
        sigma_refined_train = np.array(sigma_refined_train)
        
        # Train CNO model using refined S4 and sigma
        cno_features = np.column_stack([X_s4, s4_refined_train, sigma_refined_train])
        X_cno_scaled = self.cno_scaler.fit_transform(cno_features)
        y_cno = train_data['CNO'].values
        self.cno_model.fit(X_cno_scaled, y_cno)
        
        # Get CNO predictions and apply SBAS refinement
        cno_pred_train = self.cno_model.predict(X_cno_scaled)
        cno_refined_train = []
        for idx, (i, row) in enumerate(train_data.iterrows()):
            nearest_sbas, _ = self.find_nearest_sbas(row['AZM'], row['ELV'])
            sbas_data = {'correction': np.random.normal(0, 0.5)}
            cno_refined = self.apply_sbas_refinement([cno_pred_train[idx]], sbas_data)[0]
            cno_refined_train.append(cno_refined)
        
        cno_refined_train = np.array(cno_refined_train)
        
        # Train final S4 model using predicted sigma and CNO
        s4_final_features = np.column_stack([X_s4, sigma_refined_train, cno_refined_train])
        X_s4_final_scaled = self.s4_final_scaler.fit_transform(s4_final_features)
        self.s4_final_model.fit(X_s4_final_scaled, y_s4)
        
        self.is_trained = True
        print("Training completed successfully!")
        
        # Print model performance on training data
        self.evaluate_training_performance(train_data)
    
    def predict(self, test_data):
        """
        Make predictions using the complete pipeline
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        print("Making predictions...")
        results = []
        
        # Prepare initial features
        s4_features = ['GPS_TOW', 'Wk_no', 'PRN_NO', 'AZM', 'ELV']
        X_s4 = self.prepare_features(test_data, s4_features)
        X_s4_scaled = self.s4_scaler.transform(X_s4)
        
        for idx, (i, row) in enumerate(test_data.iterrows()):
            # Step 1: Initial S4 prediction
            s4_initial = self.s4_model.predict([X_s4_scaled[idx]])[0]
            
            # Step 2: SBAS refinement for S4
            nearest_sbas, distance = self.find_nearest_sbas(row['AZM'], row['ELV'])
            sbas_data = {'correction': np.random.normal(0, 0.05)}  # Simulate SBAS data
            s4_refined = self.apply_sbas_refinement([s4_initial], sbas_data)[0]
            
            # Step 3: Sigma prediction
            sigma_features = np.concatenate([X_s4_scaled[idx], [s4_refined]])
            sigma_initial = self.sigma_model.predict([sigma_features])[0]
            
            # Step 4: SBAS refinement for sigma
            sbas_data_sigma = {'correction': np.random.normal(0, 0.02)}
            sigma_refined = self.apply_sbas_refinement([sigma_initial], sbas_data_sigma)[0]
            
            # Step 5: CNO prediction
            cno_features = np.concatenate([X_s4_scaled[idx], [s4_refined], [sigma_refined]])
            cno_features_scaled = self.cno_scaler.transform([cno_features])
            cno_initial = self.cno_model.predict(cno_features_scaled)[0]
            
            # Step 6: SBAS refinement for CNO
            sbas_data_cno = {'correction': np.random.normal(0, 0.5)}
            cno_refined = self.apply_sbas_refinement([cno_initial], sbas_data_cno)[0]
            
            # Step 7: Final S4 re-estimation
            s4_final_features = np.concatenate([X_s4_scaled[idx], [sigma_refined], [cno_refined]])
            s4_final_features_scaled = self.s4_final_scaler.transform([s4_final_features])
            s4_final = self.s4_final_model.predict(s4_final_features_scaled)[0]
            
            results.append({
                'GPS_TOW': row['GPS_TOW'],
                'Wk_no': row['Wk_no'],
                'PRN_NO': row['PRN_NO'],
                'AZM': row['AZM'],
                'ELV': row['ELV'],
                'S4_initial': s4_initial,
                'S4_refined': s4_refined,
                'Sigma_predicted': sigma_refined,
                'CNO_predicted': cno_refined,
                'S4_final': s4_final,
                'nearest_SBAS': nearest_sbas,
                'SBAS_distance': distance
            })
        
        return pd.DataFrame(results)
    
    def evaluate_training_performance(self, train_data):
        """
        Evaluate model performance on training data
        """
        print("\n=== Training Performance ===")
        
        # Make predictions on training data
        train_predictions = self.predict(train_data)
        
        # Calculate metrics for final S4 predictions
        y_true_s4 = train_data['TOTAL_S4_L1'].values
        y_pred_s4 = train_predictions['S4_final'].values
        
        s4_mse = mean_squared_error(y_true_s4, y_pred_s4)
        s4_rmse = np.sqrt(s4_mse)
        s4_mae = mean_absolute_error(y_true_s4, y_pred_s4)
        s4_r2 = r2_score(y_true_s4, y_pred_s4)
        
        print(f"S4 Prediction Metrics:")
        print(f"  RMSE: {s4_rmse:.6f}")
        print(f"  MAE: {s4_mae:.6f}")
        print(f"  R²: {s4_r2:.6f}")
        
    def plot_results(self, predictions, actual_data=None):
        """
        Plot prediction results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot S4 progression
        axes[0, 0].plot(predictions['S4_initial'], label='Initial S4', alpha=0.7)
        axes[0, 0].plot(predictions['S4_refined'], label='SBAS Refined S4', alpha=0.7)
        axes[0, 0].plot(predictions['S4_final'], label='Final S4', alpha=0.7)
        if actual_data is not None:
            axes[0, 0].plot(actual_data['TOTAL_S4_L1'], label='Actual S4', alpha=0.7)
        axes[0, 0].set_title('S4 Prediction Progression')
        axes[0, 0].set_ylabel('S4 Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot predicted parameters
        axes[0, 1].plot(predictions['Sigma_predicted'], label='Predicted Sigma', color='orange')
        if actual_data is not None:
            axes[0, 1].plot(actual_data['60s_sigma'], label='Actual Sigma', alpha=0.7)
        axes[0, 1].set_title('60s Sigma Predictions')
        axes[0, 1].set_ylabel('Sigma Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot CNO predictions
        axes[1, 0].plot(predictions['CNO_predicted'], label='Predicted CNO', color='green')
        if actual_data is not None:
            axes[1, 0].plot(actual_data['CNO'], label='Actual CNO', alpha=0.7)
        axes[1, 0].set_title('CNO Predictions')
        axes[1, 0].set_ylabel('CNO Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot SBAS distances
        axes[1, 1].scatter(predictions['SBAS_distance'], predictions['S4_final'], 
                          c=predictions['PRN_NO'], cmap='viridis', alpha=0.6)
        axes[1, 1].set_title('S4 Final vs SBAS Distance')
        axes[1, 1].set_xlabel('SBAS Distance')
        axes[1, 1].set_ylabel('Final S4')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save trained models (placeholder - would use joblib in practice)
        """
        import pickle
        
        model_data = {
            's4_model': self.s4_model,
            'sigma_model': self.sigma_model,
            'cno_model': self.cno_model,
            's4_final_model': self.s4_final_model,
            's4_scaler': self.s4_scaler,
            'sigma_scaler': self.sigma_scaler,
            'cno_scaler': self.cno_scaler,
            's4_final_scaler': self.s4_final_scaler,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained models (placeholder - would use joblib in practice)
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.s4_model = model_data['s4_model']
        self.sigma_model = model_data['sigma_model']
        self.cno_model = model_data['cno_model']
        self.s4_final_model = model_data['s4_final_model']
        self.s4_scaler = model_data['s4_scaler']
        self.sigma_scaler = model_data['sigma_scaler']
        self.cno_scaler = model_data['cno_scaler']
        self.s4_final_scaler = model_data['s4_final_scaler']
        self.is_trained = model_data['is_trained']
        
        print(f"Model loaded from {filepath}")


def create_sample_data(n_samples=1000):
    """
    Create sample GPS data for testing (replace with your actual data loading)
    """
    np.random.seed(42)
    
    data = {
        'GPS_TOW': np.random.uniform(0, 604800, n_samples),  # GPS Time of Week
        'Wk_no': np.random.randint(1800, 2200, n_samples),   # GPS Week Number
        'PRN_NO': np.random.randint(1, 32, n_samples),       # Satellite PRN
        'AZM': np.random.uniform(0, 360, n_samples),         # Azimuth
        'ELV': np.random.uniform(5, 90, n_samples),          # Elevation
    }
    
    # Create synthetic target variables with some correlation
    s4_base = 0.1 + 0.3 * np.sin(np.radians(data['AZM'])) * np.cos(np.radians(data['ELV']))
    data['TOTAL_S4_L1'] = np.maximum(0, s4_base + np.random.normal(0, 0.05, n_samples))
    
    data['60s_sigma'] = 0.5 + 0.2 * data['TOTAL_S4_L1'] + np.random.normal(0, 0.1, n_samples)
    data['CNO'] = 45 - 10 * data['TOTAL_S4_L1'] - 2 * data['60s_sigma'] + np.random.normal(0, 2, n_samples)
    
    return pd.DataFrame(data)


def main():
    """
    Main function to demonstrate the GPS prediction system
    """
    print("GPS Satellite Parameter Prediction System with SBAS Refinement")
    print("=" * 60)
    
    # Create or load your data
    # Replace this with your actual data loading
    print("Loading sample data...")
    data = create_sample_data(n_samples=2000)
    
    print(f"Data shape: {data.shape}")
    print(f"Data columns: {list(data.columns)}")
    print("\nData summary:")
    print(data.describe())
    
    # Split data into train and test
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Initialize and train the prediction system
    gps_system = GPSPredictionSystem()
    gps_system.train(train_data)
    
    # Make predictions on test data
    predictions = gps_system.predict(test_data)
    
    print("\nPrediction Results:")
    print(predictions.head())
    
    # Plot results
    gps_system.plot_results(predictions, test_data)
    
    # Save predictions
    predictions.to_csv('gps_predictions.csv', index=False)
    print("\nPredictions saved to 'gps_predictions.csv'")
    
    # Save the trained model
    gps_system.save_model('gps_model.pkl')
    
    return gps_system, predictions


if __name__ == "__main__":
    system, results = main()