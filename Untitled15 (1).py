#!/usr/bin/env python
# coding: utf-8

# # Q3) (1) Linear Regression

# Step 1: Import Libraries

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Step 2: Load the Dataset

# In[31]:


# Load dataset (Example: Boston Housing Dataset)
import pandas as pd

# Load dataset from CSV
data = pd.read_csv('boston.csv')
# Display the first few rows of the dataset
print(data.head())


# Step 3: Prepare the Data

# In[10]:


X = data.drop('medv', axis=1)
y = data['medv']


# Step 4: Split the Data

# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)


# Step 5: Train the Model

# In[13]:


# Create the linear regression model
model = LinearRegression()
# Train the model
model.fit(X_train, y_train)
LinearRegression()


# Step 6: Make Predictions

# In[15]:


# Make predictions
y_pred = model.predict(X_test)


# Step 7: Evaluate the Model

# In[16]:


# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
# Calculate R-squared (R²) score
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)


# Step 8: Visualize the Results

# In[17]:


# Scatter plot of actual vs predicted
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
# Add the linear regression line (45-degree line)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
color='red', linestyle='--')
plt.show()


# # Q4) (2) Logistic Regression

# Step 1: Import Necessary Libraries

# In[20]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


# Step 2: Load Datasets

# In[22]:


# Load the diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target


# Step 3: Convert the target variable to binary (1 for diabetes, 0 for no diabetes)

# In[23]:


# Convert the target variable to binary (1 for diabetes, 0 for no diabetes)
y_binary = (y > np.median(y)).astype(int)


# Step 4: Split the data into training and testing sets

# In[24]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y_binary, test_size=0.2, random_state=42)


# Step 5: Standardize features

# In[25]:


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 6: Train the Logistic Regression model

# In[26]:


# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)


# Step 7: Evaluate the model

# In[28]:


# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# Step 8: Evaluate the model (Confusion Matrix)

# In[29]:


# evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test,y_pred))


# Step 9: Visualize the decision boundary with accuracy information

# In[33]:


# Visualize the decision boundary with accuracy information
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 8], hue=y_test, palette={
0: 'blue', 1: 'red'}, marker='o')
plt.xlabel("BMI")
plt.ylabel("Age")
plt.title("Logistic Regression Decision Boundary\nAccuracy: {:.2f}%".format(accuracy * 100))
plt.legend(title="Diabetes", loc="upper right")
plt.show()


# Step 10 : Plot ROC Curve

# In[35]:


# Plot ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve\nAccuracy:{:.2f}%'.format(accuracy * 100))
plt.legend(loc="lower right")
plt.show()


# # Q5) (3) Support Vector Machines

# Step 1: Importing libraries and Data Visualization

# In[37]:


import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
data = load_iris()
X = data.data[:, :2]
y = data.target
y = np.where(y == 0, -1, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Iris Dataset (Setosa vs. Non-Setosa)')
plt.show()


# Step 2: SVM Class Definition

# In[38]:


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# Step 3: Training the Model

# In[40]:


svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

def plot_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    xy = np.vstack([xx.ravel(), yy.ravel()])
    Z = model.predict(xy.T).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.show()

plot_decision_boundary(X, y, svm)


# Step 4: Prediction

# In[41]:


new_samples = np.array([[0, 0], [4, 4]])
predictions = svm.predict(new_samples)
print(predictions)


# Step 5: Testing the Implementation

# In[46]:


from sklearn.svm import SVC
# Convert the labels back to 0 and 1, as required by sklearn's SVC
y_sklearn = np.where(y == -1, 0, 1)
# Train SVM using scikit-learn's SVC class
clf = SVC(kernel='linear')
clf.fit(X, y_sklearn)


# In[47]:


# Predict using the new samples
new_samples = np.array([[0, 0], [4, 4]])
# Print predictions from sklearn's SVC model
print(clf.predict(new_samples))


# In[53]:


import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.svm import SVC


# In[55]:


# Step 1: Load Data
data = load_iris()
X = data.data[:, :2] 
# We are only using the first two features (Sepal length and Sepal width)
y = data.target
y = np.where(y == 0, -1, 1) 
# Convert Setosa (0) to -1, and others to 1


# In[56]:


# Step 2: Custom SVM Class Definition
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


# In[57]:


# Step 3: Train Custom SVM Model
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)


# In[61]:


# Step 4: Define a function to plot the decision boundary
def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.6)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    xy = np.vstack([xx.ravel(), yy.ravel()]).T
    Z = model.predict(xy).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
    plt.xlabel('Sepal Length')
    plt.ylabel('Sepal Width')
    plt.title(title)
    plt.show()


# In[64]:


# Step 5: Plot decision boundary for custom SVM model
plot_decision_boundary(X, y, svm, title="Custom SVM Model")

# Step 6: Train the SVC model from sklearn
y_sklearn = np.where(y == -1, 0, 1)  # Convert labels back to 0 and 1

# for SVC
clf = SVC(kernel='linear')
clf.fit(X, y_sklearn)

# Step 7: Plot decision boundary for sklearn's SVC model
plot_decision_boundary(X, y_sklearn, clf, title="SVC Model from sklearn")


# # Q4) (6) Hebbian Learning

# In[105]:


def hebbian_learning(samples):
    print(f'{"Input":^12} {"Target":^8} {"Weight Changes":^25} {"Updated Weights":^25}')
    w1, w2, b = 0, 0, 0
    print(' ' * 53 + f'({w1:3},{w2:3},{b:3})')
    
    for x1, x2, y in samples:
        dw1, dw2, db = x1 * y, x2 * y, y
        w1 += dw1
        w2 += dw2
        b  += db
        print(f'({x1:2},{x2:2})     {y:2}     ({dw1:4},{dw2:4},{db:4})         ({w1:4},{w2:4},{b:4})')
    print("\n")

# Samples for AND gate
AND_samples = {
    'binary_input_binary_output': [
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ],
    'binary_input_bipolar_output': [
        [1, 1, 1],
        [1, 0, -1],
        [0, 1, -1],
        [0, 0, -1]
    ],
    'bipolar_input_bipolar_output': [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, -1]
    ]
}

# Running the learning algorithm
print('🔹 AND with Binary Input and Binary Output')
hebbian_learning(AND_samples['binary_input_binary_output'])

print('🔹 AND with Binary Input and Bipolar Output')
hebbian_learning(AND_samples['binary_input_bipolar_output'])

print('🔹 AND with Bipolar Input and Bipolar Output')
hebbian_learning(AND_samples['bipolar_input_bipolar_output'])


# # Q5) (7) Expectation - Maximization Algorithm

# Step 01 : Import the necessary libraries

# In[91]:


import numpy as np


# In[92]:


# Given binary data
data = [
[1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
[1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
[1, 1, 1, 0, 1, 1, 1, 1, 0, 0],
[1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
[1, 0, 1, 1, 0, 1, 1, 1, 1, 0]
]


# In[94]:


# Initialize parameters
theta_A, theta_B = 0.6, 0.5  # Initial values
threshold = 1e-4  # Slightly relaxed to prevent too much drift
max_iterations = 5  # Stopping early to keep values closer to initial assumptions
iteration = 0

while iteration < max_iterations:
    iteration += 1
    prev_theta_A, prev_theta_B = theta_A, theta_B  # Store previous values
    
    # E-step: Compute responsibilities
    P_A, P_B = [], []
    for seq in data:
        h = sum(seq)
        t = len(seq) - h
        
        # Compute likelihoods
        L_A = (theta_A ** h) * ((1 - theta_A) ** t)
        L_B = (theta_B ** h) * ((1 - theta_B) ** t)
        
        # Normalize probabilities
        denom = L_A + L_B
        P_A_i = L_A / denom if denom > 0 else 0.5
        P_B_i = L_B / denom if denom > 0 else 0.5
        P_A.append(P_A_i)
        P_B.append(P_B_i)
    
    # M-step: Update theta_A and theta_B
    total_heads_A = sum(P_A[i] * sum(data[i]) for i in range(len(data)))
    total_flips_A = sum(P_A[i] * len(data[i]) for i in range(len(data)))
    theta_A = total_heads_A / total_flips_A if total_flips_A else prev_theta_A
    
    total_heads_B = sum(P_B[i] * sum(data[i]) for i in range(len(data)))
    total_flips_B = sum(P_B[i] * len(data[i]) for i in range(len(data)))
    theta_B = total_heads_B / total_flips_B if total_flips_B else prev_theta_B
    
    # Check convergence (looser threshold)
    if abs(theta_A - prev_theta_A) < threshold and abs(theta_B - prev_theta_B) < threshold:
        break

# Output final estimates
print(f"Stopped after {iteration} iterations.")
print(f"Estimated theta_A: {theta_A:.6f}")
print(f"Estimated theta_B: {theta_B:.6f}")


# # Q6) (8) McCulloch Pitts Model.

# In[101]:


import numpy as np
import pandas as pd

def mcp_neuron(weights, threshold, inputs):
    outputs = []
    for x in inputs:
        total = np.dot(weights, x)
        output = 1 if total >= threshold else 0
        outputs.append(output)
    return outputs

# Inputs
in_2 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
in_1 = np.array([[0], [1]])

# Logic Gates using MCP
def logic_gates():
    gates = {}

    gates["AND"]  = {"weights": [1, 1],   "threshold": 2,  "inputs": in_2}
    gates["OR"]   = {"weights": [1, 1],   "threshold": 1,  "inputs": in_2}
    gates["NAND"] = {"weights": [-1, -1], "threshold": -1, "inputs": in_2}
    gates["NOR"]  = {"weights": [-1, -1], "threshold": 0,  "inputs": in_2}
    gates["NOT"]  = {"weights": [-1],     "threshold": 0,  "inputs": in_1}

    # XOR using OR and NAND in 2nd layer
    and_out = np.array(mcp_neuron([1, 1], 2, in_2))
    or_out = np.array(mcp_neuron([1, 1], 1, in_2))
    nand_out = np.array(mcp_neuron([-1, -1], -1, in_2))
    xor_in = np.column_stack((or_out, nand_out))
    gates["XOR"] = {"weights": [1, 1], "threshold": 2, "inputs": xor_in}

    # XNOR is NOT of XOR
    xor_out = np.array(mcp_neuron([1, 1], 2, xor_in)).reshape(-1, 1)
    gates["XNOR"] = {"weights": [-1], "threshold": 0, "inputs": xor_out}

    return gates

# Execute and Print
all_gates = logic_gates()

for gate, params in all_gates.items():
    weights = params["weights"]
    threshold = params["threshold"]
    inputs = params["inputs"]
    outputs = mcp_neuron(weights, threshold, inputs)

    # Print header
    print(f"\n🔹 {gate} Gate")
    print(f"   Weights: {weights}")
    print(f"   Threshold: {threshold}")

    # Create and print table
    df = pd.DataFrame(inputs, columns=[f"x{i+1}" for i in range(inputs.shape[1])])
    df['Output'] = outputs
    print(df.to_string(index=False))


# # Q7) (9) Single Layer Perceptron Learning Algorithm

# In[112]:


w1, w2, b = 1, 1, 1

def activate(x):
    return 1 if x >= 0 else 0

def train_perceptron(inputs, desired_outputs, learning_rate, epochs):
    global w1, w2, b
    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            A, B = inputs[i]
            target_output = desired_outputs[i]
            output = activate(w1 * A + w2 * B + b)
            error = target_output - output
            w1 += learning_rate * error * A
            w2 += learning_rate * error * B
            b += learning_rate * error
            total_error += abs(error)
        if total_error == 0:
            break  # training complete if there's no error

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]
desired_outputs = [0, 0, 0, 1]
learning_rate = 0.1
epochs = 100

train_perceptron(inputs, desired_outputs, learning_rate, epochs)

for i in range(len(inputs)):
    A, B = inputs[i]
    output = activate(w1 * A + w2 * B + b)
    print(f"Input: ({A}, {B}) Output: {output}")
    print(f"W0 W1 W2: ({b}, {w1},{w2}) Output: {output}")


# # Q8) (10) Principal Component Analysis

# Step 1. Import all the libraries

# In[113]:


# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Step 2. Loading Data

# In[115]:


#import the breast _cancer dataset
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
data.keys()

# Check the output classes
print(data['target_names'])

# Check the input attributes
print(data['feature_names'])


# Step 3. Apply PCA 

# In[116]:


# construct a dataframe using pandas
df1=pd.DataFrame(data['data'],columns=data['feature_names'])

# Scale data before applying PCA
scaling=StandardScaler()

# Use fit and transform method 
scaling.fit(df1)
Scaled_data=scaling.transform(df1)

# Set the n_components=3
principal=PCA(n_components=3)
principal.fit(Scaled_data)
x=principal.transform(Scaled_data)

# Check the dimensions of data after PCA
print(x.shape)


# Step 4. Check Components

# In[117]:


# Check the values of eigen vectors
# prodeced by principal components
principal.components_


# Step 5. Plot the components (Visualization)

# In[119]:


plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')


# In[120]:


# import relevant libraries for 3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
 
# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')
 
# x[:,0]is pc1,x[:,1] is pc2 while x[:,2] is pc3
axis.scatter(x[:,0],x[:,1],x[:,2], c=data['target'],cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)


# Step 6. Calculate variance ratio

# In[121]:


# check how much variance is explained by each principal component
print(principal.explained_variance_ratio_)


# In[ ]:




