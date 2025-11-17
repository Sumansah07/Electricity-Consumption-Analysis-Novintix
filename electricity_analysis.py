import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

class ElectricityAnalysis:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_clean = None
        self.forecast_model = None
        self.anomaly_model = None
        self.cluster_model = None
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.filepath, sep=';', 
                             parse_dates={'datetime': ['Date', 'Time']},
                             infer_datetime_format=True,
                             low_memory=False,
                             na_values=['?', ''])
        
        # Convert columns to numeric
        numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                       'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        for col in numeric_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        self.df.set_index('datetime', inplace=True)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print(f"Date range: {self.df.index.min()} to {self.df.index.max()}")
        return self.df
    
    # ==================== TASK 1: EDA ====================
    def task1_eda(self):
        """Exploratory Data Analysis"""
        print("\n" + "="*60)
        print("TASK 1: EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Missing values analysis
        print("\nMissing Values Analysis:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'Missing': missing, 'Percentage': missing_pct})
        print(missing_df[missing_df['Missing'] > 0])
        
        # Basic statistics
        print("\nBasic Statistics for Global_active_power:")
        print(self.df['Global_active_power'].describe())
        
        # Plot 1: Time-series trend with missing values highlighted
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Full time series (sample for visibility)
        sample_data = self.df['Global_active_power'].resample('D').mean()
        axes[0].plot(sample_data.index, sample_data.values, linewidth=0.8, color='blue')
        axes[0].set_title('Daily Average Global Active Power - Full Timeline', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Power (kW)')
        axes[0].grid(True, alpha=0.3)
        
        # Identify abnormal readings (outliers)
        Q1 = self.df['Global_active_power'].quantile(0.25)
        Q3 = self.df['Global_active_power'].quantile(0.75)
        IQR = Q3 - Q1
        outliers = (self.df['Global_active_power'] < (Q1 - 3*IQR)) | (self.df['Global_active_power'] > (Q3 + 3*IQR))
        print(f"\nAbnormal readings detected: {outliers.sum()} ({(outliers.sum()/len(self.df)*100):.2f}%)")
        
        # Plot 2: Hourly patterns
        self.df['hour'] = self.df.index.hour
        hourly_pattern = self.df.groupby('hour')['Global_active_power'].agg(['mean', 'std'])
        axes[1].plot(hourly_pattern.index, hourly_pattern['mean'], marker='o', linewidth=2, markersize=6, color='green')
        axes[1].fill_between(hourly_pattern.index, 
                            hourly_pattern['mean'] - hourly_pattern['std'],
                            hourly_pattern['mean'] + hourly_pattern['std'],
                            alpha=0.3, color='green')
        axes[1].set_title('Hourly Consumption Pattern (Mean ± Std)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Power (kW)')
        axes[1].set_xticks(range(0, 24))
        axes[1].grid(True, alpha=0.3)
        
        # Identify high and low usage periods
        high_usage_hours = hourly_pattern.nlargest(6, 'mean').index.tolist()
        low_usage_hours = hourly_pattern.nsmallest(6, 'mean').index.tolist()
        print(f"\nHigh-usage hours: {sorted(high_usage_hours)}")
        print(f"Low-usage hours: {sorted(low_usage_hours)}")
        
        # Plot 3: Daily patterns
        self.df['day_of_week'] = self.df.index.dayofweek
        daily_pattern = self.df.groupby('day_of_week')['Global_active_power'].mean()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        axes[2].bar(range(7), daily_pattern.values, color='orange', alpha=0.7)
        axes[2].set_title('Average Consumption by Day of Week', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Day of Week')
        axes[2].set_ylabel('Power (kW)')
        axes[2].set_xticks(range(7))
        axes[2].set_xticklabels(days, rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('task1_eda_analysis.png', dpi=300, bbox_inches='tight')
        print("\n[SUCCESS] EDA plots saved as 'task1_eda_analysis.png'")
        plt.close()
        
        # Clean data for subsequent tasks
        self.df_clean = self.df.dropna(subset=['Global_active_power']).copy()
        print(f"\nCleaned dataset: {self.df_clean.shape[0]} rows")
        
    # ==================== TASK 2: TIME-SERIES FORECASTING ====================
    def task2_forecasting(self):
        """Time-series forecasting using Random Forest"""
        print("\n" + "="*60)
        print("TASK 2: TIME-SERIES FORECASTING")
        print("="*60)
        
        # Prepare windowed features
        print("\nPreparing time-series features...")
        df_model = self.df_clean[['Global_active_power']].copy()
        
        # Create lag features (past 24 hours)
        for lag in [1, 2, 3, 6, 12, 24]:
            df_model[f'lag_{lag}'] = df_model['Global_active_power'].shift(lag)
        
        # Rolling statistics
        df_model['rolling_mean_12'] = df_model['Global_active_power'].shift(1).rolling(window=12).mean()
        df_model['rolling_std_12'] = df_model['Global_active_power'].shift(1).rolling(window=12).std()
        df_model['rolling_mean_24'] = df_model['Global_active_power'].shift(1).rolling(window=24).mean()
        
        # Time-based features
        df_model['hour'] = df_model.index.hour
        df_model['day_of_week'] = df_model.index.dayofweek
        df_model['month'] = df_model.index.month
        df_model['is_weekend'] = (df_model.index.dayofweek >= 5).astype(int)
        
        # Drop NaN values created by lag features
        df_model.dropna(inplace=True)
        
        # Train-test split (80-20 temporal split)
        split_idx = int(len(df_model) * 0.8)
        train = df_model.iloc[:split_idx]
        test = df_model.iloc[split_idx:]
        
        X_train = train.drop('Global_active_power', axis=1)
        y_train = train['Global_active_power']
        X_test = test.drop('Global_active_power', axis=1)
        y_test = test['Global_active_power']
        
        print(f"Training set: {len(train)} samples")
        print(f"Test set: {len(test)} samples")
        
        # Train Random Forest model
        print("\nTraining Random Forest model...")
        self.forecast_model = RandomForestRegressor(n_estimators=100, max_depth=20, 
                                                    min_samples_split=10, random_state=42, n_jobs=-1)
        self.forecast_model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.forecast_model.predict(X_train)
        y_pred_test = self.forecast_model.predict(X_test)
        
        # Evaluation metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        print("\n--- Model Performance ---")
        print(f"Train RMSE: {train_rmse:.4f} kW")
        print(f"Test RMSE: {test_rmse:.4f} kW")
        print(f"Train MAE: {train_mae:.4f} kW")
        print(f"Test MAE: {test_mae:.4f} kW")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.forecast_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 5 Important Features:")
        print(feature_importance.head())
        
        # Visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Predicted vs Actual (sample of test set for clarity)
        sample_size = min(1000, len(test))
        sample_indices = test.index[-sample_size:]
        axes[0].plot(sample_indices, y_test[-sample_size:], label='Actual', linewidth=1.5, alpha=0.7)
        axes[0].plot(sample_indices, y_pred_test[-sample_size:], label='Predicted', linewidth=1.5, alpha=0.7)
        axes[0].set_title('Predicted vs Actual Global Active Power (Test Set Sample)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Datetime')
        axes[0].set_ylabel('Power (kW)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Scatter plot
        axes[1].scatter(y_test, y_pred_test, alpha=0.3, s=1)
        axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
        axes[1].set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Actual Power (kW)')
        axes[1].set_ylabel('Predicted Power (kW)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('task2_forecasting_results.png', dpi=300, bbox_inches='tight')
        print("\n[SUCCESS] Forecasting plots saved as 'task2_forecasting_results.png'")
        plt.close()
        
        return test, y_pred_test
    
    # ==================== TASK 3: UNSUPERVISED LEARNING ====================
    def task3_unsupervised(self):
        """Anomaly detection and clustering"""
        print("\n" + "="*60)
        print("TASK 3: UNSUPERVISED LEARNING")
        print("="*60)
        
        # --- Anomaly Detection ---
        print("\n--- Anomaly Detection ---")
        df_anomaly = self.df_clean[['Global_active_power', 'Global_reactive_power', 
                                     'Voltage', 'Global_intensity']].copy()
        
        # Use Isolation Forest
        self.anomaly_model = IsolationForest(contamination=0.01, random_state=42)
        anomalies = self.anomaly_model.fit_predict(df_anomaly)
        
        self.df_clean['anomaly'] = anomalies
        anomaly_count = (anomalies == -1).sum()
        print(f"Anomalies detected: {anomaly_count} ({(anomaly_count/len(df_anomaly)*100):.2f}%)")
        
        # --- Clustering on Daily Profiles ---
        print("\n--- Clustering Daily Consumption Profiles ---")
        
        # Create daily consumption profiles
        daily_profiles = self.df_clean.groupby(self.df_clean.index.date).agg({
            'Global_active_power': ['mean', 'std', 'min', 'max', 'sum'],
            'Global_intensity': 'mean',
            'Voltage': 'mean'
        })
        daily_profiles.columns = ['_'.join(col).strip() for col in daily_profiles.columns.values]
        daily_profiles.dropna(inplace=True)
        
        # Standardize features
        scaler = StandardScaler()
        daily_profiles_scaled = scaler.fit_transform(daily_profiles)
        
        # Determine optimal clusters using elbow method
        inertias = []
        silhouette_scores = []
        K_range = range(2, 8)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(daily_profiles_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(daily_profiles_scaled, kmeans.labels_))
        
        # Use 4 clusters (good balance)
        optimal_k = 4
        self.cluster_model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        clusters = self.cluster_model.fit_predict(daily_profiles_scaled)
        daily_profiles['cluster'] = clusters
        
        print(f"\nOptimal number of clusters: {optimal_k}")
        print(f"Silhouette Score: {silhouette_score(daily_profiles_scaled, clusters):.3f}")
        
        # Cluster characteristics
        print("\n--- Cluster Characteristics ---")
        for i in range(optimal_k):
            cluster_data = daily_profiles[daily_profiles['cluster'] == i]
            print(f"\nCluster {i} ({len(cluster_data)} days):")
            print(f"  Avg Daily Consumption: {cluster_data['Global_active_power_mean'].mean():.2f} kW")
            print(f"  Avg Daily Total: {cluster_data['Global_active_power_sum'].mean():.2f} kWh")
            print(f"  Peak Power: {cluster_data['Global_active_power_max'].mean():.2f} kW")
            
            # Categorize cluster
            avg_consumption = cluster_data['Global_active_power_mean'].mean()
            if avg_consumption < 1.0:
                category = "Low-use days"
            elif avg_consumption < 2.5:
                category = "Medium-use days"
            else:
                category = "High-use days"
            print(f"  Category: {category}")
        
        # Visualization
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Anomalies in time series
        ax1 = fig.add_subplot(gs[0, :])
        sample_data = self.df_clean['Global_active_power'].resample('H').mean().iloc[:2000]
        sample_anomalies = self.df_clean['anomaly'].resample('H').min().iloc[:2000]
        ax1.plot(sample_data.index, sample_data.values, linewidth=0.8, label='Normal', color='blue')
        anomaly_points = sample_data[sample_anomalies == -1]
        ax1.scatter(anomaly_points.index, anomaly_points.values, color='red', s=20, label='Anomaly', zorder=5)
        ax1.set_title('Anomaly Detection in Power Consumption', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Datetime')
        ax1.set_ylabel('Power (kW)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Elbow curve
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
        ax2.set_title('Elbow Method for Optimal K', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Number of Clusters')
        ax2.set_ylabel('Inertia')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Silhouette scores
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(K_range, silhouette_scores, marker='o', linewidth=2, markersize=8, color='green')
        ax3.set_title('Silhouette Score by K', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Number of Clusters')
        ax3.set_ylabel('Silhouette Score')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cluster visualization (2D projection)
        ax4 = fig.add_subplot(gs[2, 0])
        scatter = ax4.scatter(daily_profiles['Global_active_power_mean'], 
                             daily_profiles['Global_active_power_max'],
                             c=clusters, cmap='viridis', alpha=0.6, s=30)
        ax4.set_title('Clusters: Mean vs Max Daily Power', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Mean Daily Power (kW)')
        ax4.set_ylabel('Max Daily Power (kW)')
        plt.colorbar(scatter, ax=ax4, label='Cluster')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cluster distribution
        ax5 = fig.add_subplot(gs[2, 1])
        cluster_counts = daily_profiles['cluster'].value_counts().sort_index()
        ax5.bar(cluster_counts.index, cluster_counts.values, color='coral', alpha=0.7)
        ax5.set_title('Number of Days per Cluster', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Number of Days')
        ax5.grid(True, alpha=0.3, axis='y')
        
        plt.savefig('task3_unsupervised_learning.png', dpi=300, bbox_inches='tight')
        print("\n[SUCCESS] Unsupervised learning plots saved as 'task3_unsupervised_learning.png'")
        plt.close()
    
    # ==================== TASK 4: RULE-BASED AI ====================
    def task4_rule_based_ai(self, test_data, predictions):
        """Rule-based consumption category generator"""
        print("\n" + "="*60)
        print("TASK 4: RULE-BASED AI - CONSUMPTION CATEGORY GENERATOR")
        print("="*60)
        
        def categorize_usage(power):
            """Categorize power consumption"""
            if power < 1.0:
                return "Low Usage"
            elif power < 3.0:
                return "Medium Usage"
            else:
                return "High Usage"
        
        def generate_suggestion(category, power):
            """Generate rule-based suggestions"""
            suggestions = {
                "Low Usage": "[OK] Excellent! Your consumption is efficient. Continue monitoring to maintain this level.",
                "Medium Usage": "[WARNING] Moderate consumption detected. Consider identifying high-power appliances and optimizing their usage during peak hours.",
                "High Usage": "[ALERT] High consumption alert! Review your appliance usage, check for inefficient devices, and consider load shifting to off-peak hours."
            }
            return suggestions[category]
        
        # Apply to predictions
        categories = [categorize_usage(p) for p in predictions]
        
        # Statistics
        category_counts = pd.Series(categories).value_counts()
        print("\n--- Prediction Category Distribution ---")
        for cat, count in category_counts.items():
            print(f"{cat}: {count} hours ({count/len(categories)*100:.1f}%)")
        
        # Example outputs
        print("\n--- Example Outputs ---")
        sample_indices = [0, len(predictions)//3, 2*len(predictions)//3, -1]
        
        for idx in sample_indices:
            power = predictions[idx]
            category = categorize_usage(power)
            suggestion = generate_suggestion(category, power)
            timestamp = test_data.index[idx]
            
            print(f"\n{'='*50}")
            print(f"Timestamp: {timestamp}")
            print(f"Predicted Power: {power:.3f} kW")
            print(f"Category: {category}")
            print(f"Suggestion: {suggestion}")
        
        # Visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Time series with categories
        sample_size = min(500, len(predictions))
        sample_pred = predictions[-sample_size:]
        sample_time = test_data.index[-sample_size:]
        sample_cat = categories[-sample_size:]
        
        colors = {'Low Usage': 'green', 'Medium Usage': 'orange', 'High Usage': 'red'}
        for i, (time, pred, cat) in enumerate(zip(sample_time, sample_pred, sample_cat)):
            axes[0].scatter(time, pred, color=colors[cat], s=10, alpha=0.6)
        
        axes[0].set_title('Predicted Power with Usage Categories', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Datetime')
        axes[0].set_ylabel('Power (kW)')
        axes[0].axhline(y=1.0, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Low threshold')
        axes[0].axhline(y=3.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='High threshold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Category distribution
        axes[1].bar(category_counts.index, category_counts.values, 
                   color=[colors[cat] for cat in category_counts.index], alpha=0.7)
        axes[1].set_title('Usage Category Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Category')
        axes[1].set_ylabel('Number of Hours')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('task4_rule_based_ai.png', dpi=300, bbox_inches='tight')
        print("\n[SUCCESS] Rule-based AI plots saved as 'task4_rule_based_ai.png'")
        plt.close()
        
        print("\n" + "="*60)
        print("ALL TASKS COMPLETED SUCCESSFULLY!")
        print("="*60)

def main():
    # Initialize analysis
    filepath = r"d:\internship (Vive coding platform)\Novintix Assmnt\Electricity Consumption Analysis\household_power_consumption.txt"
    
    print("="*60)
    print("ELECTRICITY CONSUMPTION ANALYSIS")
    print("AI/ML Intern Assignment")
    print("="*60)
    
    analyzer = ElectricityAnalysis(filepath)
    
    # Load data
    analyzer.load_data()
    
    # Task 1: EDA
    analyzer.task1_eda()
    
    # Task 2: Forecasting
    test_data, predictions = analyzer.task2_forecasting()
    
    # Task 3: Unsupervised Learning
    analyzer.task3_unsupervised()
    
    # Task 4: Rule-based AI
    analyzer.task4_rule_based_ai(test_data, predictions)
    
    print("\n[SUCCESS] All visualizations saved successfully!")
    print("[SUCCESS] Check the generated PNG files for detailed results.")

if __name__ == "__main__":
    main()
