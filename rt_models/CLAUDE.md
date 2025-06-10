# CLAUDE.md - RT Models Directory

This directory contains pre-trained retention time (RT) prediction models for JMod.

## Navigation
- [← Back to Main Documentation](../CLAUDE.md)
- [→ Models Module](../src/models/CLAUDE.md)
- [→ RT Alignment Module](../src/rt_alignment.py)

## Overview

The RT models are CNN-based neural networks trained to predict peptide retention times for different labeling methods. These models are essential for the RT alignment step in JMod's pipeline.

## Model Structure

### Model Naming Convention
`iRT_CNN_model_{TAG}_{DATE}_{INDEX}/`

Where:
- `TAG`: Labeling method (LF, mTRAQ, DiEthyl)
- `DATE`: Training date (e.g., 09182024)
- `INDEX`: Ensemble member (0-4)

### Special Models
`iRT_TransferLearning_Tag6_updated_{DATE}_{INDEX}/`
- Transfer learning models for Tag6 labeling
- Built on top of base models with additional training

## Available Models

### Label-Free (LF)
- `iRT_CNN_model_LF_09182024_0` through `_4`
- Trained on unmodified peptides
- Default for `--tag ''` or no tag specified

### mTRAQ
- `iRT_CNN_model_mTRAQ_09182024_0` through `_4`
- Trained on mTRAQ-labeled peptides
- Used with `--tag mTRAQ`

### DiEthyl
- `iRT_CNN_model_DiEthyl_11052024_0` through `_4`
- Trained on diethyl-labeled peptides
- Used with `--tag diethyl`

### Tag6 (Transfer Learning)
- `iRT_TransferLearning_Tag6_updated_05072025_0` through `_4`
- Special transfer learning models
- Used with `--tag tag6`

## Model Format

Each model directory contains:
- `saved_model.pb`: Model architecture and weights
- `keras_metadata.pb`: Keras-specific metadata
- `fingerprint.pb`: Model fingerprint for verification
- `variables/`: Directory containing model weights
  - `variables.data-00000-of-00001`: Weight values
  - `variables.index`: Weight index

## Usage in JMod

### Model Selection
The RT alignment module (`src/rt_alignment.py`) selects models based on:
1. The `--tag` parameter from command line
2. Automatic tag detection from peptide modifications
3. Default to label-free if no tag specified

### Ensemble Prediction
- All 5 models (0-4) are loaded for the selected tag
- Predictions are averaged across the ensemble
- This improves prediction robustness

### Integration Flow
1. **Library Loading**: Peptide sequences extracted from spectral library
2. **Model Loading**: Appropriate models loaded based on tag
3. **Prediction**: RT predicted for all library peptides
4. **Alignment**: Predictions used to align experimental RTs

## Model Performance

### Accuracy Expectations
- Label-free: ±2-3 minutes typical error
- Modified peptides: ±3-4 minutes typical error
- Performance depends on chromatography consistency

### Factors Affecting Performance
1. **Chromatography Differences**: Models assume specific gradient
2. **Modification Coverage**: Rare modifications may predict poorly
3. **Peptide Length**: Very short/long peptides less accurate

## Technical Details

### Input Format
- Peptide sequences encoded as numeric arrays
- One-hot encoding for amino acids
- Fixed sequence length with padding

### Architecture
- Convolutional Neural Network (CNN)
- Multiple convolutional layers
- Global pooling
- Dense layers for RT prediction

### Training
- Trained on large-scale empirical RT data
- Tag-specific datasets for each model type
- Transfer learning for new modifications

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   ValueError: Could not find SavedModel
   ```
   - Ensure model directory exists
   - Check TensorFlow version compatibility

2. **Wrong Tag Models**
   ```
   Warning: No RT model found for tag 'custom'
   ```
   - Use supported tags: '', 'mTRAQ', 'diethyl', 'tag6'
   - Train custom models if needed

3. **Poor RT Predictions**
   - Verify correct tag parameter
   - Check if peptides have expected modifications
   - Consider chromatography differences

## Adding New Models

To add models for a new tag:

1. **Train Models**: Use same architecture with tag-specific data
2. **Name Consistently**: Follow naming convention
3. **Create Ensemble**: Train 5 models (0-4)
4. **Update Code**: Add tag handling in `rt_alignment.py`

## Dependencies

- TensorFlow 2.x: Model loading and inference
- NumPy: Array operations
- Model files must be in SavedModel format

## Related Documentation
- [Main JMod Documentation](../CLAUDE.md)
- [RT Alignment Code](../src/rt_alignment.py)
- [Models Module](../src/models/CLAUDE.md)
- [Configuration Options](../src/config.py)