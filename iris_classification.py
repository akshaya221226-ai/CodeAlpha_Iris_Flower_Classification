# ============================================================
# CODEALPHA TASK 1: IRIS FLOWER CLASSIFICATION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("           IRIS FLOWER CLASSIFICATION")
print("=" * 70)

file_path = r"C:\Users\Hp\Downloads\archive (6)\Iris.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")


# ============================================================
# 2. DISPLAY DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n" + "=" * 70)
print("DATA CLEANING")
print("=" * 70)

# Remove unnecessary Id column
if "Id" in df.columns:
    df = df.drop("Id", axis=1)

# Remove completely empty rows
df = df.dropna(how="all")

# Remove rows containing missing values
df = df.dropna()

print("\nData after cleaning:")
print(df.head())

print("\nDataset shape after cleaning:")
print(df.shape)


# ============================================================
# 4. SEPARATE FEATURES AND TARGET
# ============================================================

X = df[
    [
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
]

y = df["Species"]


# ============================================================
# 5. ENCODE TARGET LABELS
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("\nSpecies classes:")
print(label_encoder.classes_)


# ============================================================
# 6. SPLIT DATA INTO TRAINING AND TESTING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 7. TRAIN MACHINE LEARNING MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MACHINE LEARNING MODEL")
print("=" * 70)

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. CALCULATE ACCURACY
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nModel Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 10. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 12. SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)

sample_data = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [6.0, 2.9, 4.5, 1.5],
    [6.5, 3.0, 5.2, 2.0]
])

sample_predictions = model.predict(sample_data)

for i in range(len(sample_data)):
    predicted_species = label_encoder.inverse_transform(
        [sample_predictions[i]]
    )[0]

    print(
        f"\nSample {i + 1}: "
        f"{predicted_species}"
    )


# ============================================================
# 13. GRAPH 1 - SPECIES DISTRIBUTION
# ============================================================

plt.figure(figsize=(8, 6))

sns.countplot(
    data=df,
    x="Species"
)

plt.title("Iris Species Distribution")
plt.xlabel("Species")
plt.ylabel("Number of Flowers")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig(
    "01_species_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 14. GRAPH 2 - SEPAL LENGTH VS WIDTH
# ============================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="SepalLengthCm",
    y="SepalWidthCm",
    hue="Species",
    s=70
)

plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")

plt.tight_layout()

plt.savefig(
    "02_sepal_length_vs_width.png",
    dpi=300
)

plt.show()


# ============================================================
# 15. GRAPH 3 - PETAL LENGTH VS WIDTH
# ============================================================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="PetalLengthCm",
    y="PetalWidthCm",
    hue="Species",
    s=70
)

plt.title("Petal Length vs Petal Width")
plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

plt.tight_layout()

plt.savefig(
    "03_petal_length_vs_width.png",
    dpi=300
)

plt.show()


# ============================================================
# 16. GRAPH 4 - CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")

plt.tight_layout()

plt.savefig(
    "04_confusion_matrix.png",
    dpi=300
)

plt.show()


# ============================================================
# 17. GRAPH 5 - FEATURE COMPARISON
# ============================================================

plt.figure(figsize=(10, 6))

df.boxplot(
    column=[
        "SepalLengthCm",
        "SepalWidthCm",
        "PetalLengthCm",
        "PetalWidthCm"
    ]
)

plt.title("Iris Flower Feature Distribution")
plt.ylabel("Measurement (cm)")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "05_feature_distribution.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual": label_encoder.inverse_transform(y_test),
    "Predicted": label_encoder.inverse_transform(y_pred)
})

results.to_csv(
    "iris_predictions.csv",
    index=False
)


# ============================================================
# 19. SAVE REPORT
# ============================================================

with open(
    "iris_classification_report.txt",
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "IRIS FLOWER CLASSIFICATION REPORT\n"
    )

    report.write(
        "=" * 60 + "\n\n"
    )

    report.write(
        f"Total dataset samples: {len(df)}\n"
    )

    report.write(
        f"Training samples: {len(X_train)}\n"
    )

    report.write(
        f"Testing samples: {len(X_test)}\n"
    )

    report.write(
        f"Model: Logistic Regression\n"
    )

    report.write(
        f"Accuracy: {accuracy * 100:.2f}%\n\n"
    )

    report.write(
        "Species:\n"
    )

    for species in label_encoder.classes_:
        report.write(
            f"- {species}\n"
        )

    report.write(
        "\nConclusion:\n"
    )

    report.write(
        "The Logistic Regression model successfully "
        "classifies Iris flowers into three species "
        "using sepal and petal measurements. The model "
        "was evaluated using test data and accuracy, "
        "classification report, and confusion matrix.\n"
    )


# ============================================================
# 20. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("             ANALYSIS COMPLETED")
print("=" * 70)

print("\nFiles created:")

print("1. 01_species_distribution.png")
print("2. 02_sepal_length_vs_width.png")
print("3. 03_petal_length_vs_width.png")
print("4. 04_confusion_matrix.png")
print("5. 05_feature_distribution.png")
print("6. iris_predictions.csv")
print("7. iris_classification_report.txt")

print("\nIris Flower Classification completed successfully!")