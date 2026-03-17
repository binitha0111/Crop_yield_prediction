# Crop_Yield_Prediction

End-to-end machine learning system to predict crop yield using environmental and agricultural data such as rainfall, temperature, and pesticide usage.

---
## Dataset
- FAO Crop Yield & Pesticide Data  
- World Bank Rainfall & Temperature Data  

https://www.fao.org/faostat/  
https://climateknowledgeportal.worldbank.org  

---

## Methodology
- Data cleaning and merging (multi-source datasets)  
- Feature engineering (interaction features, encoding, scaling)  
- Model training using Random Forest and XGBoost  

---

## Model Performance
- R² Score: **0.97**  
- Random Forest selected for stable performance  

---

## Application
- Flask REST API for predictions  
- Streamlit dashboard for visualization  

---
## 3D Visualization
- Interactive 3D globe to visualize crop suitability across countries  
- Displays region-wise best crops based on environmental and yield factors  
- Helps in understanding global agricultural patterns and decision-making  
