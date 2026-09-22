# 🧬 CRISPR Off-Target Prediction Using Deep Learning

## 📌 Project Overview

CRISPR-Cas9 is a powerful gene-editing technology that enables targeted modification of DNA. However, CRISPR systems can sometimes bind to unintended DNA sequences, known as **off-target sites**, which can affect the safety and reliability of gene-editing applications.

This project develops a **Deep Learning-based CRISPR Off-Target Prediction system** that analyzes guide RNA (sgRNA) and potential off-target DNA sequences to estimate the probability of off-target activity.

The project combines:

* 🧬 DNA sequence information
* 🔬 Sequence mismatch features
* 📊 GC-content features
* 🤖 Convolutional Neural Networks (CNN)
* ⚖️ Focal Loss for class imbalance
* 🌐 Streamlit-based prediction interface

---

## 🎯 Objectives

The main objectives of this project are:

1. Predict potential CRISPR-Cas9 off-target activity.
2. Process and encode DNA sequences for machine learning.
3. Extract meaningful sequence-level features.
4. Build a hybrid CNN model using sequence and manually engineered features.
5. Handle severe class imbalance using class weighting and focal loss.
6. Evaluate model performance using appropriate classification metrics.
7. Develop an interactive web application for predictions.

---

## 🏗️ Project Architecture

```text
                 CRISPR Input
                      │
          ┌───────────┴───────────┐
          │                       │
       sgRNA                Off-target DNA
          │                       │
          └───────────┬───────────┘
                      │
                Preprocessing
                      │
          ┌───────────┴───────────┐
          │                       │
     One-Hot Encoding       Manual Features
          │                  ┌────┴────┐
          │                  │         │
          │              Mismatch   GC Content
          │                  │         │
          └──────────┬───────┴─────────┘
                     │
                  CNN Model
                     │
              Feature Fusion
                     │
              Dense Layers
                     │
              Prediction
                     │
          Off-Target Probability
                     │
             ┌───────┴────────┐
             │                │
          Low Risk          High Risk
```

---

## 📂 Dataset

The project uses two CRISPR-related datasets:

### 1. CIRCLE-seq Dataset

File:

```text
CIRCLE_seq.csv
```

### 2. GUIDE-seq Dataset

File:

```text
GUIDE-Seq.csv
```

The datasets contain information related to guide RNA sequences, potential off-target sequences, labels, and sequencing/read information.

The datasets were cleaned and standardized before combining them for model development.

---

## 🔄 Data Preprocessing

The preprocessing pipeline includes:

1. Loading the datasets.
2. Standardizing column names.
3. Combining the datasets.
4. Removing missing values.
5. Processing sgRNA and off-target sequences.
6. One-hot encoding DNA sequences.
7. Calculating mismatch counts.
8. Calculating GC content.
9. Preparing sequence and manual feature inputs.
10. Splitting the data into training and testing sets.

### DNA One-Hot Encoding

Each DNA nucleotide is represented as a four-dimensional vector:

```text
A → [1, 0, 0, 0]
T → [0, 1, 0, 0]
G → [0, 0, 1, 0]
C → [0, 0, 0, 1]
```

This allows DNA sequences to be processed by the CNN.

---

## 🧠 Feature Engineering

Two main types of features are used.

### Sequence Features

The sgRNA and off-target sequences are combined and converted into a 2D one-hot encoded representation.

```text
Sequence → One-Hot Encoding → CNN
```

### Manual Features

The model also uses:

* **Mismatch Count**
* **GC Content**

These features are combined with CNN-extracted sequence features before the final prediction layers.

---

## 🤖 Deep Learning Model

The project uses a **Hybrid CNN Architecture**.

### Model Pipeline

```text
DNA Sequence
     ↓
One-Hot Encoding
     ↓
Conv1D
     ↓
MaxPooling
     ↓
Conv1D
     ↓
MaxPooling
     ↓
Flatten
     ↓
Sequence Features
          \
           → Concatenate → Dense → Dropout → Dense → Sigmoid
          /
Manual Features
```

### CNN Configuration

The model uses:

* Conv1D: 64 filters
* Conv1D: 128 filters
* MaxPooling layers
* Dense layer: 128 neurons
* Dropout: 0.4
* Dense layer: 64 neurons
* Sigmoid output layer

The final output represents the predicted probability of off-target activity.

---

## ⚖️ Handling Class Imbalance

CRISPR off-target datasets can contain a highly imbalanced distribution between positive and negative examples.

To address this problem, the project uses:

### Class Weighting

Balanced class weights are calculated during training so that the minority class receives greater importance.

### Focal Loss

Focal Loss is used to focus training on difficult and minority-class examples.

The model uses:

```text
Gamma = 2
Alpha = 0.25
```

This helps reduce the effect of the large class imbalance during training.

---

## 📊 Model Evaluation

Because accuracy can be misleading for highly imbalanced datasets, additional evaluation metrics were considered.

### Evaluation Metrics

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score

The model achieved approximately:

```text
ROC-AUC : 0.978
PR-AUC  : 0.517
```

A Logistic Regression baseline using manually engineered features was also evaluated for comparison.

```text
Logistic Regression PR-AUC : 0.062
```

These results show the importance of sequence-based information compared with using only basic manually engineered features.

> Note: Model performance depends on the dataset split, preprocessing, threshold selection, and training configuration.

---

## 🌐 Streamlit Web Application

The project includes an interactive Streamlit application.

Users can enter:

```text
sgRNA Sequence
Off-target Sequence
```

The application then:

1. Validates the sequences.
2. Encodes the DNA sequences.
3. Calculates mismatch count.
4. Calculates GC content.
5. Passes the features to the trained CNN model.
6. Generates an off-target probability.
7. Displays a risk classification.

### Example

```text
Input
───────────────
sgRNA:        DNA sequence
Off-target:   DNA sequence

              ↓

      Hybrid CNN Model

              ↓

Prediction Probability
              ↓

     Low / High Risk
```

---

## 🛠️ Technologies Used

| Technology         | Purpose                     |
| ------------------ | --------------------------- |
| Python             | Programming                 |
| NumPy              | Numerical computation       |
| Pandas             | Data preprocessing          |
| Scikit-learn       | ML utilities and evaluation |
| TensorFlow / Keras | Deep Learning               |
| CNN                | DNA sequence modeling       |
| Streamlit          | Web application             |
| Jupyter Notebook   | Experimentation             |
| Git & GitHub       | Version control             |

---

## 📁 Project Structure

```text
crispr-off-target-prediction/
│
├── app.py
├── backend_api.py
├── frontent_api.py
├── PDF_Report.py
│
├── NEW CRISPR.ipynb
│
├── CIRCLE_seq.csv
├── GUIDE-Seq.csv
│
├── hybrid_model.h5
├── best_threshold.npy
│
├── requirements.txt
├── README.md
│
└── .gitignore
```

Virtual environments such as `venv/` and `tf_env/` are excluded from the repository.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/abhilesh200/crispr-off-target-prediction.git
```

### 2. Move into the project directory

```bash
cd crispr-off-target-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔬 How the Prediction Works

The application receives an sgRNA and candidate off-target sequence.

The system calculates:

```text
Mismatch Count
GC Content
Sequence Encoding
```

These inputs are passed to the trained hybrid CNN model.

The model produces a probability:

```text
0 → Lower predicted risk
1 → Higher predicted risk
```

A predefined threshold stored in:

```text
best_threshold.npy
```

is used to convert the probability into the displayed risk category.

---

## 📈 Future Improvements

Possible future improvements include:

* Testing additional CRISPR datasets.
* Adding more sequence features.
* Exploring Transformer-based DNA models.
* Hyperparameter optimization.
* Cross-dataset validation.
* Explainable AI for sequence-level predictions.
* Deploying the Streamlit application online.
* Adding batch prediction for multiple sequences.
* Improving model calibration.
* Comparing CNN performance with RNN, Transformer, and traditional ML models.

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes**.

The prediction output should not be considered a definitive biological or clinical assessment. Experimental validation and domain-specific analysis are required before drawing biological conclusions.

---

## 👨‍💻 Author

**Abhilesh Kumar**

Data Science | Machine Learning | Deep Learning

GitHub:

https://github.com/abhilesh200

---

## ⭐ Project Highlights

* 🧬 CRISPR-Cas9 off-target prediction
* 🤖 Hybrid CNN deep learning model
* 🔬 DNA sequence one-hot encoding
* 📊 Mismatch and GC-content features
* ⚖️ Focal Loss for class imbalance
* 📈 ROC-AUC and PR-AUC evaluation
* 🌐 Interactive Streamlit application
* 🐍 Python + TensorFlow/Keras
* 🚀 End-to-end machine learning workflow
