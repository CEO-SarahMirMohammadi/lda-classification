from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Generate a synthetic classification dataset
X, y = make_classification(
    n_samples=2000,
    n_features=6,
    n_informative=2,
    n_redundant=0,
    random_state=0,
    shuffle=False
)

# Display the generated dataset
print("Dataset:")
print(X)

print("\nDataset shape:")
print(X.shape)


# Create the Linear Discriminant Analysis classifier
clf = LinearDiscriminantAnalysis()

# Train the model
clf.fit(X, y)

print("\nModel trained successfully.")


# Make a prediction using 6 features
sample = [[0, 0, 0, 0, 0, 0]]

prediction = clf.predict(sample)

print("\nPrediction:")
print(prediction)