# FDA-final
https://colab.research.google.com/drive/1kIONk1nZ5kuuzOeOsjdTGt_sPkyqFtKI
https://www.canva.com/design/DAGkQLAESAc/3jtMqr9kCaxCfJyAyaLPTA/edit?utm_content=DAGkQLAESAc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("/content/bs140513_032310.csv")

# Drop uninformative columns
df.drop(columns=["zipcodeOri", "zipMerchant"], inplace=True)

# Clean and encode categorical variables
for col in ["customer", "age", "gender", "merchant", "category"]:
    df[col] = df[col].str.replace("'", "")
    df[col] = LabelEncoder().fit_transform(df[col])

# Drop rows with missing target values if any
df = df.dropna(subset=["fraud"])

# Features and target
X = df.drop("fraud", axis=1)
y = df["fraud"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE on training data only
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=200, random_state=42),
    "Isolation Forest": IsolationForest(contamination=0.01, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    print(f"\n{name}")
    
    if name == "Isolation Forest":
        model.fit(X_train_scaled)  # No SMOTE for unsupervised model
        preds = model.predict(X_test_scaled)
        preds = [1 if p == -1 else 0 for p in preds]
    else:
        model.fit(X_train_resampled, y_train_resampled)  # SMOTE data
        preds = model.predict(X_test_scaled)

    print(classification_report(y_test, preds))
    print("AUC Score:", roc_auc_score(y_test, preds))
