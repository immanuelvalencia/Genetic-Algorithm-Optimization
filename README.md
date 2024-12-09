
# Diabetes Risk Assessment Using Genetic Algorithm for Feature Selection

This project implements a Decision Support System for assessing diabetes risk using predictive modeling. A Genetic Algorithm (GA) is employed to optimize feature selection, minimizing the number of inputs while maximizing the predictive accuracy of the model. The system uses various machine learning classifiers, including k-Nearest Neighbors (KNN), Random Forest, Logistic Regression, Support Vector Machine (SVM), and Decision Tree, to evaluate the effectiveness of selected features.

## Authors

- **Engr. Immanuel Jose C. Valencia**
- **Engr. Sherwin Lagonoy**
- **Dr. Robert Kerwin C. Billones**

For inquiries or collaborations, feel free to reach out to any of the authors.

## Features
- **Feature Optimization**: Uses Genetic Algorithm to select the optimal subset of features.
- **Predictive Modeling**: Supports multiple classifiers (KNN, Random Forest, Logistic Regression, SVM, Decision Tree).
- **Scalable Design**: Easily adaptable to other datasets and classifiers.
- **Open Source**: Code and methodology are shared for reproducibility and further research.

---

## Files
- **`diabetes_train.csv`**: Training dataset.
- **`diabetes_test.csv`**: Testing dataset.
- **`main.py`**: Python script containing the implementation of feature optimization and predictive modeling.
- **`README.md`**: Documentation for the project.

---

## Dependencies
Ensure the following Python libraries are installed:
- `numpy`
- `pandas`
- `sklearn`
- `geneticalgorithm`

Install the required packages using:
```bash
pip install numpy pandas scikit-learn geneticalgorithm
```

---

## Data Format
Both the training (`diabetes_train.csv`) and testing (`diabetes_test.csv`) datasets should have the following columns:
1. **gender**
2. **age**
3. **hypertension**
4. **heart_disease**
5. **smoking_history**
6. **bmi**
7. **HbA1c_level**
8. **blood_glucose_level**
9. **diabetes** (Target variable: 1 for diabetes, 0 for no diabetes)

---

## How to Run the Code
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Ensure the training (`diabetes_train.csv`) and testing (`diabetes_test.csv`) datasets are in the same directory as the script.

3. Open the script (`main.py`) to:
   - Configure the classifier you wish to use by uncommenting the respective function in the `f(X)` function.
   - Adjust Genetic Algorithm parameters in the `algorithm_param` dictionary if needed.

4. Run the script:
   ```bash
   python main.py
   ```

5. The Genetic Algorithm will optimize the feature selection, and the selected feature set along with the model's accuracy will be displayed after the run.

---

## Genetic Algorithm Parameters
The Genetic Algorithm uses the following parameters for feature selection:
| **Parameter**                     | **Value**         |
|------------------------------------|-------------------|
| Max Number of Iterations           | 100               |
| Population Size                    | 100               |
| Mutation Probability               | 0.1               |
| Elitism Ratio                      | 0                 |
| Crossover Probability              | 0.5               |
| Parents Portion                    | 0.3               |
| Crossover Type                     | Uniform           |
| Max Iterations Without Improvement | 30                |

---

## Classifiers
The script supports the following classifiers for predictive modeling:
1. **K-Nearest Neighbors (KNN)**
2. **Random Forest**
3. **Logistic Regression**
4. **Support Vector Machine (SVM)**
5. **Decision Tree**

You can switch between classifiers by modifying the `f(X)` function in the script.

---

## Outputs
- The selected feature set optimized by the Genetic Algorithm.
- Accuracy of the chosen classifier on the test dataset.

---

## Contributing
Contributions are welcome! If you encounter issues or have ideas for improvement, feel free to create a pull request or open an issue in the repository.

---

## License
This project is open-source and available under the [MIT License](LICENSE).

---

## Acknowledgments
This study supports the use of open data and open-source methodologies to advance research in healthcare. By optimizing input selection and sharing the code, the project aims to foster collaboration and innovation in decision support systems for diabetes risk assessment.
