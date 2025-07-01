# Inflation Forecasting in Crisis Times: A Comprehensive Bayesian and Machine Learning Approach

A systematic comparison of nine forecasting models for inflation prediction during crisis periods, with particular emphasis on Bayesian Dynamic Linear Models (DLMs). This project demonstrates the superior performance of adaptive Bayesian methods over traditional econometric approaches during periods of high volatility and structural change.

## 🎯 Key Results

**Best Performing Model**: MLP
- **RMSE**: 1.3639 (out-of-sample)

### Model Performance Ranking (Out-of-Sample)
1. **Multi-Layer Perception - 1.3639** 🏆
2. ARIMA - 1.4637
3. Bayesian DLM (SMC) - 1.4724
4. OLS + RFE - 1.6489
5. OLS - 1.6639
6. XGBoost - 1.7556
7. Random Forest - 1.7665
8. Bayesian DLM (MCMC) - 1.8404
9. Bayesian DLM (DR) - 4.0686

## 🧠 Models Implemented

### Traditional Econometric Methods
- **ARIMA(3,1,2)**: Baseline time series model with automatic order selection
- **OLS**: Linear regression with and without Recursive Feature Elimination (RFE)

### Machine Learning Approaches
- **Random Forest**: Bootstrap aggregated decision trees
- **XGBoost**: Gradient boosting with regularization
- **Multi-Layer Perceptron**: Neural network with hidden layers

### Bayesian Dynamic Linear Models (Core Innovation)
1. **MCMC Variant**: Time-varying parameters with Gaussian random walk
2. **SMC with Regime Switching**: Explicit crisis period modeling with 3× volatility multiplier
3. **Dynamic Regression**: Time-varying coefficients for multiple predictors

## 📊 Key Insights

### ✅ What Works
- **Adaptive Models Excel**: Bayesian DLMs significantly outperform static models during crisis periods
- **Uncertainty Quantification**: Bayesian approaches provide crucial confidence intervals for decision-making
- **Regime Awareness**: Models that explicitly account for structural breaks perform better
- **Simplicity Wins**: Parsimonious models often outperform complex ones in crisis scenarios

### ⚠️ Important Findings
- **Overfitting Alert**: XGBoost achieved perfect in-sample fit (RMSE: 0.0274) but poor generalization (RMSE: 1.7556)
- **Computational Trade-offs**: More complex doesn't always mean better performance
- **Crisis-Specific Dynamics**: Traditional predictor relationships break down during volatile periods

## 🗂️ Project Structure

```
inflation-forecasting/
├── dataset.csv                        
├── src/                       
│   ├── models/                  
│   ├── utils.py                
│   └── metrics.py        
├── results/                     
├── report.pdf
├── main.py              
└── README.md

```

## 🚀 Getting Started



## 📈 Methodology

### Data & Evaluation
- **Training Period**: 2000-2019 (pre-crisis)
- **Test Period**: 2020+ (COVID-19 crisis)
- **Approach**: Rolling one-step-ahead forecasting
- **Metric**: Root Mean Square Error (RMSE)

### Bayesian DLM Implementation Highlights
- **MCMC**: NUTS sampler with 2000 draws, 4 chains
- **SMC**: 5000 particles with stratified resampling
- **Convergence**: All models achieved R̂ < 1.01

## 📄 Detailed Analysis

For comprehensive methodology, mathematical formulations, diagnostic plots, and in-depth discussion of results, please refer to the **[full research report (PDF)](report.pdf)**.

The report includes:
- Complete mathematical specifications for all models
- MCMC diagnostic plots and convergence analysis
- Detailed discussion of computational considerations
- Policy implications for central banks and financial institutions
- Robustness analysis and model limitations

## 🎯 Applications

This research is directly applicable to:
- **Central Banking**: Monetary policy decisions during crisis periods
- **Risk Management**: Stress testing and scenario analysis
- **Financial Planning**: Investment strategy during volatile periods
- **Academic Research**: Methodological framework for crisis-time forecasting

## 🔬 Technical Notes

- All results are reproducible with fixed random seeds
- Minor variations (< 0.1 RMSE) may occur due to MCMC numerical precision
- Computational time: Bayesian methods ~10× slower than traditional approaches

## 📝 Citation

If you use this work in your research, please cite:
```
De Giorgi, S. (2025). Inflation Forecasting in Crisis Times: A Comprehensive 
Bayesian and Machine Learning Approach. Available at: 
https://github.com/Simo-dg/Inflation-forecasting
```
### Note: This is a non-peer-reviewed research project.

*This project demonstrates that in times of crisis, adaptive and uncertainty-aware models significantly outperform traditional static approaches for inflation forecasting.*
