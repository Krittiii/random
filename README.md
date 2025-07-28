# GPS Satellite Parameter Prediction System with SBAS Refinement

A comprehensive machine learning system for predicting GPS satellite parameters (TOTAL_S4_L1, 60s sigma, CNO) using satellite geometry data and SBAS (Satellite-Based Augmentation System) refinement.

## Overview

This system implements a multi-stage prediction pipeline:

1. **Initial S4 Prediction**: Predicts TOTAL_S4_L1 from GPS_TOW, Week_no, PRN_NO, Azimuth, and Elevation
2. **SBAS Refinement**: Refines S4 predictions using nearest SBAS satellite corrections
3. **Sigma Prediction**: Predicts 60s sigma from refined S4 values
4. **SBAS Sigma Refinement**: Further refines sigma predictions
5. **CNO Prediction**: Predicts CNO from estimated S4 and sigma values
6. **SBAS CNO Refinement**: Refines CNO predictions
7. **Final S4 Re-estimation**: Re-estimates S4 using predicted sigma and CNO values

## Features

- **Multi-stage prediction pipeline** with iterative refinement
- **SBAS integration** for improved accuracy using nearest satellite corrections
- **Comprehensive evaluation metrics** (RMSE, MAE, R²)
- **Detailed visualizations** showing prediction evolution and performance
- **Flexible data loading** with automatic column mapping and preprocessing
- **Model persistence** for saving and loading trained models
- **Outlier detection and removal**
- **Feature engineering** with derived satellite geometry features

## Installation

1. Clone or download the repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Required Data Format

Your GPS dataset should contain the following columns:

- `GPS_TOW`: GPS Time of Week (0-604800 seconds)
- `Wk_no`: GPS Week Number
- `PRN_NO`: Satellite PRN Number (1-32)
- `AZM`: Azimuth angle (0-360 degrees)
- `ELV`: Elevation angle (0-90 degrees)
- `CNO`: Carrier-to-Noise ratio (dB-Hz)
- `TOTAL_S4_L1`: S4 scintillation index
- `60s_sigma`: 60-second sigma value

The system will automatically handle common alternative column names (e.g., 'TOW' → 'GPS_TOW', 'Azimuth' → 'AZM').

## Usage

### Basic Usage

1. **Prepare your data**: Place your 1-week GPS dataset in CSV, Excel, or JSON format
2. **Update the data path**: Edit `run_prediction.py` and change `data_filepath` to your file path
3. **Run the system**:

```bash
python run_prediction.py
```

### Advanced Usage

#### Using Individual Components

```python
from gps_prediction_system import GPSPredictionSystem
from data_loader import load_your_data

# Load and preprocess data
data, stats = load_your_data('your_gps_data.csv')

# Initialize and train the system
gps_system = GPSPredictionSystem()
gps_system.train(train_data)

# Make predictions
predictions = gps_system.predict(test_data)
```

#### Loading a Pre-trained Model

```python
# Load a saved model
gps_system = GPSPredictionSystem()
gps_system.load_model('trained_gps_model.pkl')

# Make predictions on new data
new_predictions = gps_system.predict(new_data)
```

## Output Files

The system generates several output files:

- `detailed_results.csv`: Complete predictions with intermediate values and error metrics
- `gps_prediction_results.png`: Comprehensive visualization plots
- `trained_gps_model.pkl`: Saved trained model for future use

## System Architecture

### Models Used

- **S4 Initial Model**: Random Forest Regressor
- **Sigma Model**: Gradient Boosting Regressor  
- **CNO Model**: Random Forest Regressor
- **S4 Final Model**: Gradient Boosting Regressor

### SBAS Satellites

The system includes three SBAS satellites with configurable positions:
- SBAS_1: GAGAN PRN 127 (55°E)
- SBAS_2: GAGAN PRN 128 (98°E)
- SBAS_3: MSAS PRN 129 (142°E)

### Feature Engineering

The system automatically creates derived features:
- Time-based features (hour of day, day of week)
- Trigonometric satellite geometry features
- Elevation and PRN categorization

## Performance Metrics

The system evaluates predictions using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)

Performance is tracked for:
- Initial S4 predictions
- SBAS-refined S4 predictions
- Final S4 re-estimations
- Sigma predictions
- CNO predictions

## Customization

### SBAS Satellite Configuration

Modify the SBAS satellite positions in `gps_prediction_system.py`:

```python
self.sbas_satellites = {
    'SBAS_1': {'lat': your_lat, 'lon': your_lon, 'alt': 35786000},
    # Add more SBAS satellites as needed
}
```

### Model Configuration

Change the machine learning models in the `__init__` method:

```python
self.s4_model = YourPreferredRegressor()
self.sigma_model = AnotherRegressor()
# etc.
```

### SBAS Refinement Algorithm

Implement your specific SBAS correction algorithm in the `apply_sbas_refinement` method.

## File Structure

```
├── gps_prediction_system.py    # Main prediction system class
├── data_loader.py              # Data loading and preprocessing utilities
├── run_prediction.py           # Complete example script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Example Results

The system provides detailed analysis including:

- Prediction accuracy metrics for each stage
- Visualization of S4 evolution through the pipeline
- Scatter plots showing predicted vs actual values
- SBAS satellite usage statistics
- Residual analysis for model validation

## Troubleshooting

### Common Issues

1. **Missing columns**: The system will attempt automatic column mapping, but ensure your data contains the required fields
2. **Data format**: Supported formats are CSV, Excel (.xlsx), and JSON
3. **Memory usage**: For very large datasets, consider processing in chunks
4. **Model convergence**: Adjust model parameters if training doesn't converge

### Error Messages

- `Required columns missing`: Check your data format and column names
- `Model must be trained before making predictions`: Train the model first using `.train()`
- `Error loading data`: Verify file path and format

## Contributing

To extend the system:

1. Add new prediction models by inheriting from the base classes
2. Implement additional SBAS correction algorithms
3. Add new feature engineering techniques
4. Extend visualization capabilities

## License

This project is provided as-is for research and educational purposes.

## Contact

For questions or issues, please refer to the code documentation and comments within each module.