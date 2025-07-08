# 🚗 Predictive Analytics: Linear Regression on Vehicle Fuel Efficiency

## 📌 Abstract
This study investigates the factors contributing to vehicle fuel efficiency, measured in miles per gallon (MPG), using a dataset with various automotive attributes. Through data cleaning, exploratory data analysis (EDA), and model optimization, the most significant predictors of MPG were identified. Machine learning models including **Linear Regression**, **Ridge Regression**, and **Lasso Regression** were employed.

🔍 Key findings:
- **Model Year**, **Acceleration**, and **Vehicle Origin** significantly influence MPG.
- Advanced regularization techniques enhanced model interpretability and accuracy.

---

## 📖 Introduction
Fuel efficiency is a key concern due to:
- Environmental impact 🌍
- Rising fuel prices ⛽
- Regulatory demands 📜

This project aims to:
1. Identify important predictors of MPG.
2. Build and evaluate predictive models.
3. Derive actionable insights from data.

---

## 🧪 Methods

### 🔧 Data Cleaning
- **Data Inspection**: Checked for missing and non-numeric values.  
  ![Data Inspection](data_inspection.png)

- **Missing Value Handling**: Imputed using median strategy.  
  ![Missing Values](missing values1.png)

- **Outlier Removal**: Detected with boxplots and removed using IQR method.  
  ![Outlier Detection](images/outlier_detection.png)

- **Normalization**: Applied MinMaxScaler to scale numerical features.

### 📊 Exploratory Data Analysis (EDA)
- **Correlation Heatmap**  
  ![Correlation Heatmap](images/correlation_heatmap.png)

- **Pairplot Analysis**  
  ![Pairplot](images/pairplot.png)

- **Multicollinearity Check**: Removed features with high VIF values.

---

## 🎯 Feature Selection
- **Statistical Significance**: Selected features with p-value < 0.05.
- **Lasso Regression**: Automatically selected impactful features.

---

## 🤖 Model Building & Evaluation

### 📈 1. Linear Regression
- MAE = **0.245**, RMSE = **0.310**, R² = **0.78**
- Key predictors: Model Year, Acceleration, Vehicle Origin (US)

### 📘 2. Ridge Regression
- R² = **0.79** — better generalization and reduced overfitting

### 📕 3. Lasso Regression
- R² = **0.78**, fewer features used, simplifying interpretation

### 📊 Feature Importance  
![Feature Importance](images/feature_importance.png)

### 📉 Residual Analysis  
![Residual Plot](images/residuals.png)

---

## 💬 Discussion
- **Model Year**: Newer cars = higher MPG
- **Acceleration**: Better tuning = improved efficiency
- **Vehicle Origin**: US-made cars had slightly lower MPG

Regularized models (Ridge/Lasso) confirmed model robustness.

---

## ✅ Conclusion
This study effectively used machine learning and statistical techniques to uncover the most impactful features influencing MPG. Continuous innovation in vehicle design, especially post-2000 models, shows promising trends in fuel efficiency.

**Next Steps**:
- Try **non-linear models** (e.g., Random Forest, Gradient Boosting)
- Use **larger, real-world datasets**

---

## 📚 References
- [Scikit-learn](https://jmlr.org/papers/v12/pedregosa11a.html)
- [Statsmodels](https://www.statsmodels.org/)
- [Matplotlib](https://doi.org/10.1109/MCSE.2007.55)
- [Seaborn](https://doi.org/10.21105/joss.03021)

---

## 🧠 Author
**Mohammed Saif Wasay**  
*Data Analytics Graduate | Machine Learning Enthusiast | Passionate about turning data into insights*

---

> 🔗 Feel free to fork, star ⭐, or reach out with feedback or suggestions!



