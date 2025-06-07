# 🏠 House Price Prediction using Linear Regression

This project demonstrates how to build a **Machine Learning model** to predict house prices based on features like the number of bedrooms, size (in square feet), and age of the house. The model uses **Linear Regression** and includes feature scaling, evaluation metrics, and visualizations.

---

## 📁 Project Structure

```
├── house_price_prediction.py     # Main Python code
├── README.md                     # Project documentation
```

---

## 🔧 Technologies Used

- Python 3
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

---

## 📊 Dataset

The dataset used is synthetic and contains the following features:

| Feature   | Description                          |
|-----------|--------------------------------------|
| bedrooms  | Number of bedrooms in the house      |
| size      | Size of the house in square feet     |
| age       | Age of the house in years            |
| price     | Price of the house (target variable) |

---

## ⚙️ How It Works

1. **Data Preparation**  
   Creates a dataset with `bedrooms`, `size`, `age`, and `price`.

2. **Feature Scaling**  
   Applies `StandardScaler` to normalize the feature data.

3. **Model Building**  
   Uses `LinearRegression()` from `sklearn` to train the model.

4. **Evaluation**  
   Evaluates the model using:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R² Score

5. **Visualization**  
   Generates a scatter plot of actual vs predicted house prices.

6. **Prediction**  
   Predicts prices for new unseen data points.

---

## 🧪 Example Output

```
Mean Absolute Error (MAE): 13071.09
Mean Squared Error (MSE): 232305367.99
R² Score: 0.94

Predicted Prices for New Data:
[354105.58 403327.20]
```

---

## 📌 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

3. Run the script:
   ```bash
   python house_price_prediction.py
   ```

---

## 📈 Visualization Example

*Red dashed line = Ideal prediction line*

```
A scatter plot will be shown comparing actual prices (blue) with predicted prices (red).
```

---

## 📚 Learning Outcomes

- Understanding regression models for prediction tasks
- Preprocessing features using scaling
- Evaluating model performance using appropriate metrics
- Visualizing model results for better interpretation

---

## ✅ Future Improvements

- Use real-world housing datasets (e.g., from Kaggle or Zillow)
- Try other regression algorithms (Ridge, Lasso, SVR)
- Add more features (location, garage, bathroom, etc.)
- Convert into a Flask or Streamlit web app

---

## 📬 Contact

Feel free to reach out if you have questions or want to collaborate!

**Author:** *Sreyashi Saha*  
**Email:** sreyashis31@gmail.com

