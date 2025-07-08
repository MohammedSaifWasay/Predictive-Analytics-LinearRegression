# 🚗 Predictive Analytics: Linear Regression on Vehicle Fuel Efficiency

## 📌 Abstract
This project investigates the factors influencing vehicle fuel efficiency, measured in miles per gallon (MPG), using a dataset containing various automotive attributes. The analysis involves extensive data preprocessing, exploratory data analysis (EDA), and the implementation of regression models including **Linear Regression**, **Ridge Regression**, and **Lasso Regression**. The study identifies key predictors of MPG and demonstrates the effectiveness of regularization techniques in improving model performance and interpretability.

---

## 📖 Introduction
Fuel efficiency is a crucial metric in the automotive industry due to increasing environmental concerns, rising fuel prices, and tightening government regulations. The primary goals of this project are:

- To identify the automotive features that most significantly impact fuel efficiency.
- To build regression models that predict MPG using these features.
- To extract actionable insights from the data that could inform vehicle design and policy decisions.

---

## 🧪 Methods

### Data Cleaning
- Inspected the dataset for missing, non-numeric, and irregular values.
- Imputed missing numerical values using the **median**, a robust measure against outliers.
- Removed outliers using the **Interquartile Range (IQR)** method to prevent skewed model performance.
- Normalized continuous variables using **MinMaxScaler** to ensure uniform feature scaling.

### Exploratory Data Analysis (EDA)
- Performed correlation analysis to identify relationships between features.
- Analyzed pairwise feature relationships and distribution trends.
- Calculated **Variance Inflation Factor (VIF)** to detect and address multicollinearity.

### Feature Selection
- Used **p-values** from statistical modeling to identify significant predictors.
- Applied **Lasso Regression** to automatically select the most relevant features by shrinking less important ones to zero.

---

## 🤖 Model Building & Evaluation

Three regression models were developed:

1. **Linear Regression**  
   - R²: 0.78  
   - Significant predictors: model year, acceleration, vehicle origin.

2. **Ridge Regression**  
   - R²: 0.79  
   - Improved generalization and reduced overfitting via L2 regularization.

3. **Lasso Regression**  
   - R²: 0.78  
   - Simplified model by selecting only impactful features using L1 regularization.

Models were evaluated using:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R-squared (R²)**

---

## 💬 Discussion

### Key Findings:
- **Model Year**: Strong positive impact on MPG; newer vehicles tend to be more efficient.
- **Acceleration**: Positively associated with MPG, suggesting better-tuned engines.
- **Vehicle Origin**: US-made vehicles showed slightly lower MPG compared to foreign-made cars.

### Model Insights:
- **Regularization** techniques improved robustness and interpretability.
- **Lasso Regression** effectively simplified the model by removing redundant features.

---

## ✅ Conclusion
The analysis successfully identified critical predictors of vehicle fuel efficiency using regression-based machine learning models. This project demonstrates the power of combining statistical techniques with machine learning to derive practical insights from real-world data.

### Future Work:
- Explore non-linear models such as **Random Forest** or **Gradient Boosting** to capture complex patterns.
- Expand the analysis using more diverse or up-to-date datasets to improve generalizability.

---

## 📚 References
- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.*
- Seabold & Perktold (2010). *Statsmodels: Econometric and statistical modeling with Python.*
- Hunter (2007). *Matplotlib: A 2D graphics environment.*
- Waskom (2021). *Seaborn: Statistical data visualization.*
- ![Python](https://img.shields.io/badge/Python-3.8-blue?logo=python)
- 🐍 Built with: Python, Jupyter Notebook


---

## 🧠 Author
**Mohammed Saif Wasay**  
*Data Analytics Graduate | Machine Learning Enthusiast | Passionate about turning data into insights*
🔗 [Connect with me on LinkedIn](https://www.linkedin.com/in/mohammed-saif-wasay-4b3b64199/)
---
> ⭐ Feel free to fork this repo, star it if you find it helpful, and reach out if you have feedback or questions!
