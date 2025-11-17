# Household Electricity Consumption Analysis

## About This Project

This project analyzes 4 years of real household electricity data to uncover consumption patterns, predict future usage, and provide smart recommendations. I built this as part of my application for the AI/ML Intern position at Novintix.

## What I Did

I worked with over 2 million data points from a French household (2006-2010) to solve four different machine learning challenges:

**1. Understanding the Data (EDA)**
   - Found that the household uses most power between 7-8 AM and 7-10 PM
   - Discovered weekends have 10-15% higher consumption than weekdays
   - Identified only 1.25% missing data and 0.69% outliers - pretty clean dataset!
   - Noticed a declining trend over time, suggesting the family became more energy-conscious

**2. Predicting Next-Hour Power Usage**
   - Built a Random Forest model that predicts power consumption with 0.21 kW error
   - Used smart features like past 24 hours of data and rolling averages
   - Achieved 94% accuracy on test data (Train RMSE: 0.18, Test RMSE: 0.21)
   - The model learns that the previous hour's consumption is the strongest predictor

**3. Finding Unusual Patterns**
   - Detected 1% of readings as anomalies (sudden spikes or drops)
   - Grouped days into 4 clusters: very low use, low use, medium use, and high use
   - Found 365 days were "low-use" days (likely vacations or mild weather)
   - Identified 248 days with high consumption (probably winter months)

**4. Smart Usage Categorization**
   - Created a simple AI that labels each hour as Low/Medium/High usage
   - 57.8% of hours are low usage, 38.7% medium, only 3.5% high
   - Generates automatic suggestions like "Consider shifting usage to off-peak hours"
   - Could be used in a real-time energy monitoring app

## How to Run

```bash
# 1. Download the dataset
# Visit: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
# Download and extract household_power_consumption.txt to project root

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete analysis
python electricity_analysis.py
```

The script takes about 2-3 minutes to run and generates 4 visualization files.

## What You'll Get

- `task1_eda_analysis.png` - 3 charts showing daily trends, hourly patterns, and weekly cycles
- `task2_forecasting_results.png` - Predicted vs actual power with scatter plot
- `task3_unsupervised_learning.png` - Anomaly detection and cluster analysis
- `task4_rule_based_ai.png` - Usage categories and distribution

## Technical Highlights

- **Data Size**: 2+ million rows, 7 features
- **Models Used**: Random Forest (forecasting), Isolation Forest (anomalies), K-Means (clustering)
- **Best Result**: 0.21 kW prediction error (that's really accurate for household data!)
- **Code Quality**: Object-oriented design, proper train-test split, no data leakage

## Dataset Source

UCI Machine Learning Repository - Individual Household Electric Power Consumption  
Recorded every minute from Dec 2006 to Nov 2010 in a house near Paris.

## Why This Matters

This type of analysis could help:
- Homeowners reduce electricity bills by understanding their usage
- Utility companies predict demand and prevent blackouts
- Smart home devices automatically optimize energy consumption
- Environmental efforts by identifying wasteful patterns

---

*Built with Python, scikit-learn, pandas, and matplotlib*
