# Credit Card Approval Prediction
An end-to-end Machine Learning web application that predicts whether a credit card application is likely to be **Approved** or **Not Approved**, based on applicant financial and demographic information.

## 📌 Project Overview
This repository features a complete data science pipeline consisting of exploratory data analysis, data preprocessing and feature engineering, machine learning model training on historical applicant data, and a production-ready Flask web deployment. Users can input applicant details — income, employment, education, family status, and housing information — through an intuitive web interface to instantly receive a credit card approval prediction.

The application is developed using Flask and a Logistic Regression model trained on historical credit card applicant data. Users enter applicant details through the web interface, the input is preprocessed using the saved scaler and encoders, and the trained model predicts whether the application is likely to be approved or rejected. The application is deployed on Render, enabling real-time access through a web browser.

The evaluation relies on key applicant attributes:
- **Personal:** Gender, Age, Family Status, Family Members Count
- **Financial:** Annual Income, Income Type, Years Employed
- **Background:** Education Type
- **Housing & Assets:** Housing Type, Car Ownership, Property Ownership

🛠 Technologies Used
- Python
- Flask
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- HTML
- CSS

## 📊 Machine Learning Models
- Logistic Regression ✅ (Best Model)
- Random Forest Classifier
- Decision Tree Classifier

## 📈 Model Performance

| Model | Accuracy | F1-Score | Status |
|---|---|---|---|
| Logistic Regression | 86.77% | 0.929 | ✅ Selected Best Model |
| Random Forest | 86.71% | 0.929 | Alternate Model |
| Decision Tree | 76.88% | 0.865 | Baseline Model |

> **Note:** The dataset is imbalanced (~71% Approved vs. ~29% Not Approved). To ensure the model can meaningfully distinguish both outcomes rather than defaulting to the majority class, the final deployed model uses **class-weight balancing**, prioritizing fair detection of Not Approved applicants over raw accuracy alone.

## 🌐 Live Demo
**Live Application:** https://credit-card-approval-prediction-fisx.onrender.com

## 📂 Repository Structure
```
├── 1. Brainstorming & Ideation/     # Idea generation and prioritization
├── 2. Requirement Analysis/         # Problem statements, empathy map, customer journey, DFD, solution requirements
├── 3. Project Design Phase/         # ER diagram, architecture and design artifacts
├── 4. Project Planning Phase/       # Sprint/task planning documents
├── 5. Project Development Phase/    # Source code - data preprocessing, model training, Flask app
├── 6. Project Testing/              # Performance testing and test results
├── 7. Project Documentation/        # Full project documentation and screenshots
├── 8. Project Demonstration/        # Demo video and walkthrough
└── README.md                        # Project documentation
```

## ⚙️ Setup & Run Locally
1. Clone the repository:
   ```
   git clone https://github.com/Poojithamenon06/Credit_Card_Approval_Prediction
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure `model.pkl`, `scaler.pkl`, and `encoders.pkl` are present in the project root.
4. Run the application:
   ```
   python app.py
   ```
5. Open your browser at `http://127.0.0.1:5000`

## 👤 Author
**Poojitha Menon**
4th Year B.Tech, Vasireddy Venkatadri Institute of Technology
