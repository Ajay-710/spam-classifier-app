# AI-Powered Phishing & Spam Classifier

A machine learning project that classifies text messages (or emails) as either legitimate ("ham") or malicious ("spam"). This project uses Natural Language Processing (NLP) techniques and is deployed as an interactive web application using Streamlit.

  
*Replace the link above with a screenshot of your running Streamlit app!*

---

## 🚀 Live Demo

**You can try the live application here:** [Link to your deployed Streamlit App]

*(Once you deploy on Streamlit Community Cloud, come back and put the link here.)*

---

## 🛠️ Technologies & Libraries Used

- **Python**: The core programming language.
- **Scikit-learn**: For building and training the machine learning model (Naive Bayes).
- **Pandas**: For data manipulation and loading the dataset.
- **TF-IDF Vectorizer**: For converting text data into a numerical format suitable for the model.
- **Streamlit**: For creating and deploying the interactive web interface.
- **Git & GitHub**: For version control and hosting the project.

---

## ✨ Features

- **Text Classification**: Classifies any given text message into 'Spam' or 'Ham'.
- **Probability Score**: Shows the confidence score of the prediction.
- **Web Interface**: Simple and intuitive UI for easy testing.
- **Cached Model**: The ML model is trained once and cached for fast performance.

---

## ⚙️ How to Run This Project Locally

### Prerequisites
- Python 3.8+
- Git

### Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ajay-710/spam-classifier-app.git
    cd spam-classifier-app
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

The application should now be running in a new tab in your web browser!
