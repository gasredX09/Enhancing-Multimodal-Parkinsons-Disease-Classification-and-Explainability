# Running Gait EDA Notebook

This guide explains how to run the gait EDA notebook and what to expect.

## Prerequisites

1. **Python environment** with required packages:

   ```bash
   pip install numpy pandas matplotlib seaborn scipy
   ```

2. **Gait data** in one of these locations:
   - `/Users/aryansharanreddyguda/biomedAI/Multimodal-Parkinson-Disease-Prediction-With-XAI/gait_data/`
   - Or update the `GAIT_DATA_DIR` path in the notebook to point to your data folder

3. **Jupyter** installed:

   ```bash
   pip install jupyter
   ```

## File Structure Expected

The notebook expects `.txt` files in your gait data directory. Files should follow naming conventions:

- `GaCo*.txt` - Healthy controls (Gait Cognition)
- `GaPt*.txt` - Parkinson's patients (Gait Parkinson)
- `JuCo*.txt` - Other subjects (Ju prefix)

Each file should be a matrix of values with format:

```
time channel1 channel2 channel3 ... channelN
0.01  0.123    0.456    0.789   ... 0.234
0.02  0.234    0.567    0.890   ... 0.345
...
```

The notebook will automatically:

- Skip the time column (first column)
- Handle missing/malformed lines gracefully
- Extract all numeric data

## Running the Notebook

### Option 1: Jupyter Lab (Recommended)

```bash
cd /Users/aryansharanreddyguda/biomedAI/project/replica/01_eda
jupyter lab gait_eda.ipynb
```

### Option 2: Jupyter Notebook

```bash
cd /Users/aryansharanreddyguda/biomedAI/project/replica/01_eda
jupyter notebook gait_eda.ipynb
```

### Option 3: VS Code

1. Open the notebook file in VS Code
2. Click "Run All" or step through cells

## Notebook Sections

1. **Setup**: Import libraries and define configuration
2. **Utility Functions**: Helper functions for reading gait data files
3. **Load Data**: Discover and load all .txt files from disk
4. **Parse & Aggregate**: Organize data by subject
5. **File Metadata**: Create DataFrame of all files
6. **Signal Analysis**: Compute statistics per subject
7. **Distributions**: Visualize signal characteristics
8. **Time Series Plots**: Sample signals from 5 subjects
9. **Correlation Analysis**: Channel-to-channel correlations
10. **Frequency Domain**: FFT analysis and power spectra
11. **Quality Checks**: Detect NaN, Inf, outliers
12. **Preprocessing Summary**: Recommendations for next steps

## Expected Outputs

The notebook will create visualizations in:

```
/Users/aryansharanreddyguda/biomedAI/project/replica/outputs/gait_eda/
```

Files created:

- `01_signal_distributions.png` - Distribution histograms
- `02_sample_timeseries.png` - 5 sample signals
- `03_correlation_matrix.png` - Channel correlations
- `04_frequency_domain.png` - FFT analysis
- `subject_stats.csv` - Statistics by subject
- `file_metadata.csv` - Metadata for all files

## Troubleshooting

### "No .txt files found"

- Check that `GAIT_DATA_DIR` path is correct
- Verify gait data files are `.txt` format
- Try `print(os.listdir(GAIT_DATA_DIR))` to see folder contents

### Memory issues with large files

- Cell 4 loads all files into memory
- If OOM error occurs, process files in batches
- Modify loop to skip large files: `if file_size_mb > 100: continue`

### Import errors

- Make sure all required packages are installed: `pip install numpy pandas matplotlib seaborn scipy`

### Path errors on macOS

- Use forward slashes `/` in paths (not backslash `\`)
- Use `Path()` from pathlib for cross-platform compatibility
- Example: `Path('/Users/aryansharanreddyguda/....')` works on all systems

## Next Steps After EDA

1. **Preprocessing notebook** (create next):
   - Implement windowing (2-sec at 100Hz = 200 samples)
   - Apply StandardScaler normalization
   - Feature extraction

2. **Alignment notebook**:
   - Align gait features with handwriting/speech features
   - Create matching subject lists
   - Handle missing modalities

3. **Model training** (02_unimodal/03_gait/):
   - Use extracted features
   - Train classifier or clustering model
   - Save features for fusion

## Notes

- **Gait sampling rate**: Typically 100 Hz (0.01s per sample)
- **Standard window**: 2 seconds = 200 timesteps
- **Window overlap**: 50% (step by 100 samples)
- **Channels**: Usually 18+ channels (accelerometers, gyroscopes, etc.)

---

**Last updated**: 2026-02-18
