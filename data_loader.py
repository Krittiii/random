import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class GPSDataLoader:
    """
    Utility class for loading and preprocessing GPS week dataset
    """
    
    def __init__(self):
        self.required_columns = ['GPS_TOW', 'Wk_no', 'PRN_NO', 'AZM', 'ELV', 
                               'CNO', 'TOTAL_S4_L1', '60s_sigma']
    
    def load_data(self, filepath, file_format='csv'):
        """
        Load GPS data from file
        
        Parameters:
        -----------
        filepath : str
            Path to the data file
        file_format : str
            Format of the file ('csv', 'excel', 'json')
        
        Returns:
        --------
        pd.DataFrame
            Loaded and validated data
        """
        try:
            if file_format.lower() == 'csv':
                data = pd.read_csv(filepath)
            elif file_format.lower() in ['excel', 'xlsx']:
                data = pd.read_excel(filepath)
            elif file_format.lower() == 'json':
                data = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            print(f"Successfully loaded data from {filepath}")
            print(f"Data shape: {data.shape}")
            
            # Validate columns
            self.validate_data(data)
            
            # Clean and preprocess data
            data = self.preprocess_data(data)
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, data):
        """
        Validate that required columns are present
        """
        missing_cols = [col for col in self.required_columns if col not in data.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            print(f"Available columns: {list(data.columns)}")
            
            # Try to map common alternative column names
            column_mapping = {
                'TOW': 'GPS_TOW',
                'Week': 'Wk_no', 
                'Week_no': 'Wk_no',
                'PRN': 'PRN_NO',
                'Azimuth': 'AZM',
                'Elevation': 'ELV',
                'C/N0': 'CNO',
                'S4': 'TOTAL_S4_L1',
                'TOTAL_S4': 'TOTAL_S4_L1',
                'Sigma': '60s_sigma',
                'sigma_60s': '60s_sigma'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in data.columns and new_name in missing_cols:
                    data.rename(columns={old_name: new_name}, inplace=True)
                    print(f"Mapped column '{old_name}' to '{new_name}'")
        
        # Check for remaining missing columns
        still_missing = [col for col in self.required_columns if col not in data.columns]
        if still_missing:
            raise ValueError(f"Required columns missing: {still_missing}")
    
    def preprocess_data(self, data):
        """
        Clean and preprocess the GPS data
        """
        print("Preprocessing data...")
        
        # Remove rows with missing values in critical columns
        critical_cols = ['GPS_TOW', 'Wk_no', 'PRN_NO', 'AZM', 'ELV']
        data = data.dropna(subset=critical_cols)
        
        # Handle missing values in target variables
        for col in ['CNO', 'TOTAL_S4_L1', '60s_sigma']:
            if col in data.columns:
                # Fill missing values with median
                median_val = data[col].median()
                data[col].fillna(median_val, inplace=True)
        
        # Remove outliers (basic approach)
        data = self.remove_outliers(data)
        
        # Ensure proper data types
        data['GPS_TOW'] = pd.to_numeric(data['GPS_TOW'], errors='coerce')
        data['Wk_no'] = pd.to_numeric(data['Wk_no'], errors='coerce')
        data['PRN_NO'] = pd.to_numeric(data['PRN_NO'], errors='coerce')
        data['AZM'] = pd.to_numeric(data['AZM'], errors='coerce')
        data['ELV'] = pd.to_numeric(data['ELV'], errors='coerce')
        
        # Validate ranges
        data = data[
            (data['GPS_TOW'] >= 0) & (data['GPS_TOW'] <= 604800) &  # Valid GPS TOW
            (data['Wk_no'] > 0) &  # Valid week number
            (data['PRN_NO'] >= 1) & (data['PRN_NO'] <= 32) &  # Valid GPS PRN
            (data['AZM'] >= 0) & (data['AZM'] <= 360) &  # Valid azimuth
            (data['ELV'] >= 0) & (data['ELV'] <= 90)  # Valid elevation
        ]
        
        print(f"Data shape after preprocessing: {data.shape}")
        return data.reset_index(drop=True)
    
    def remove_outliers(self, data, method='iqr', threshold=1.5):
        """
        Remove outliers from the dataset
        """
        numeric_cols = ['CNO', 'TOTAL_S4_L1', '60s_sigma']
        
        for col in numeric_cols:
            if col in data.columns:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers_before = len(data)
                    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                    outliers_removed = outliers_before - len(data)
                    
                    if outliers_removed > 0:
                        print(f"Removed {outliers_removed} outliers from {col}")
        
        return data
    
    def add_derived_features(self, data):
        """
        Add derived features that might be useful for prediction
        """
        print("Adding derived features...")
        
        # Time-based features
        data['Hour_of_Day'] = (data['GPS_TOW'] % 86400) / 3600  # Hour of day
        data['Day_of_Week'] = ((data['GPS_TOW'] // 86400) % 7)  # Day of week
        
        # Satellite geometry features
        data['Sin_AZM'] = np.sin(np.radians(data['AZM']))
        data['Cos_AZM'] = np.cos(np.radians(data['AZM']))
        data['Sin_ELV'] = np.sin(np.radians(data['ELV']))
        data['Cos_ELV'] = np.cos(np.radians(data['ELV']))
        
        # Elevation categories
        data['ELV_Category'] = pd.cut(data['ELV'], 
                                    bins=[0, 15, 30, 60, 90], 
                                    labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        
        # PRN groups (different satellite blocks have different characteristics)
        data['PRN_Group'] = pd.cut(data['PRN_NO'], 
                                 bins=[0, 8, 16, 24, 32], 
                                 labels=['Group1', 'Group2', 'Group3', 'Group4'])
        
        return data
    
    def get_data_statistics(self, data):
        """
        Generate comprehensive statistics about the dataset
        """
        print("\n=== DATA STATISTICS ===")
        print(f"Total records: {len(data)}")
        print(f"Date range: Week {data['Wk_no'].min()} to Week {data['Wk_no'].max()}")
        print(f"Unique satellites (PRN): {data['PRN_NO'].nunique()}")
        print(f"PRN range: {data['PRN_NO'].min()} to {data['PRN_NO'].max()}")
        
        print("\nParameter ranges:")
        for col in ['AZM', 'ELV', 'CNO', 'TOTAL_S4_L1', '60s_sigma']:
            if col in data.columns:
                print(f"  {col}: {data[col].min():.3f} to {data[col].max():.3f} "
                      f"(mean: {data[col].mean():.3f}, std: {data[col].std():.3f})")
        
        print("\nMissing values:")
        missing = data.isnull().sum()
        if missing.sum() > 0:
            for col, count in missing.items():
                if count > 0:
                    print(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
        else:
            print("  No missing values")
        
        return {
            'total_records': len(data),
            'unique_prns': data['PRN_NO'].nunique(),
            'week_range': (data['Wk_no'].min(), data['Wk_no'].max()),
            'parameter_stats': data.describe()
        }


def load_your_data(filepath):
    """
    Convenience function to load your GPS data
    Replace this with your actual data loading logic
    """
    loader = GPSDataLoader()
    
    # Load the data
    data = loader.load_data(filepath)
    
    # Add derived features
    data = loader.add_derived_features(data)
    
    # Get statistics
    stats = loader.get_data_statistics(data)
    
    return data, stats


if __name__ == "__main__":
    # Example usage
    print("GPS Data Loader Utility")
    print("=" * 30)
    
    # You would replace this with your actual file path
    # data, stats = load_your_data("your_gps_data.csv")
    
    # For demonstration, create sample data
    from gps_prediction_system import create_sample_data
    
    sample_data = create_sample_data(n_samples=5000)
    sample_data.to_csv('sample_gps_data.csv', index=False)
    
    # Load the sample data using our loader
    data, stats = load_your_data('sample_gps_data.csv')
    
    print(f"\nLoaded data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")