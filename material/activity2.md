# Activity 2:

There are 2 parts in this activity. 

---

## **Part 1: Introduction to Keras**

This lab introduces the basic workflow for building and training a neural network using Keras, a deep learning library in Python. Each step is explained clearly so that beginners can follow along and understand the purpose behind the code.


### **1. Import Required Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
from keras import Input
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import pandas as pd
```

**Explanation:**

- `numpy`: Used for numerical operations, such as arrays and mathematical calculations.
- `matplotlib.pyplot`: Used for plotting graphs to visualize data and training results.
- `pandas`: Used for working with datasets in table (DataFrame) format.
- `keras.Input`: Used to define the input shape of the model.
- `tensorflow.keras.Sequential`: Used to build a neural network model layer by layer.
- `tensorflow.keras.layers.Dense`: Used to create fully connected (dense) layers in the network.
- `sklearn.model_selection.train_test_split`: Used to split the dataset into training and testing sets.


### **2. Load the Datasets**

```python
cereal_data = pd.read_csv("https://github.com/ML-Course-2025/session5/raw/refs/heads/main/material/datasets/cereal.csv")
concrete_data = pd.read_csv("https://github.com/ML-Course-2025/session5/raw/refs/heads/main/material/datasets/concrete.csv")
```

**Explanation:**

- `pd.read_csv(...)` loads CSV (comma-separated values) files from URLs.
- Two datasets are loaded:
  - `cereal_data` contains information about different cereals and their nutrition values.
  - `concrete_data` contains information about the ingredients used in making concrete and the resulting compressive strength.


### **3. Preprocess the Data**

```python
cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

concrete_features = concrete_data.drop(columns=['CompressiveStrength'])
concrete_target = concrete_data['CompressiveStrength']
```

**Explanation:**

- For both datasets, we separate the **features** (input variables) from the **target** (the value we want to predict).
- In the cereal dataset:
  - Features are nutritional values.
  - Target is the cereal’s rating.
- In the concrete dataset:
  - Features are ingredient quantities.
  - Target is the compressive strength of the concrete.


### **4. Split Data into Training and Testing Sets**

```python
cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(
    cereal_features, cereal_target, test_size=0.2, random_state=42)

concrete_X_train, concrete_X_test, concrete_y_train, concrete_y_test = train_test_split(
    concrete_features, concrete_target, test_size=0.2, random_state=42)
```

**Explanation:**

- `train_test_split(...)` splits the data into training and testing sets.
- 80% of the data is used for training, and 20% is used for testing.
- `random_state=42` ensures the split is the same every time the code is run (for reproducibility).


### **5. Define the Neural Network Model**

```python
model = Sequential()
model.add(Input(shape=(cereal_X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

**Explanation:**

- A **Sequential model** allows us to stack layers one after another.
- `Input(shape=(...))`: Defines the shape of the input data. `cereal_X_train.shape[1]` gives the number of features.
- `Dense(64, activation='relu')`: Adds a dense (fully connected) layer with 64 neurons and ReLU activation function.
- `Dense(32, activation='relu')`: Adds another dense layer with 32 neurons.
- `Dense(1)`: Output layer with one neuron (since this is a regression task predicting a single numeric value).


### **6. Compile the Model**

```python
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
```

**Explanation:**

- `optimizer='adam'`: The optimizer adjusts the weights during training. Adam is commonly used and performs well in most cases.
- `loss='mean_squared_error'`: The loss function measures how far predictions are from actual values. MSE is suitable for regression tasks.
- `metrics=['mae']`: Mean Absolute Error is used to monitor performance during training.


### **7. Train the Model**

```python
history_cereal = model.fit(
    cereal_X_train, cereal_y_train,
    epochs=10,
    batch_size=64,
    validation_data=(cereal_X_test, cereal_y_test),
    verbose=1
)
```

**Explanation:**

- `.fit(...)` trains the model on the training data.
- `epochs=10`: The model will go through the entire training data 10 times.
- `batch_size=64`: The model updates weights every 64 samples.
- `validation_data=(...)`: Evaluates model performance on test data during training.
- `verbose=1`: Displays training progress.


### **8. Visualize the Training History**

```python
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Explanation:**

- We use `matplotlib.pyplot` to create a line plot.
- The graph shows how the loss (error) changes over time during training.
- Lower validation loss indicates better performance on unseen data.
- Plotting both training and validation loss helps detect overfitting (when the model performs well on training data but poorly on test data).

> [!TIP]
> Here's the [complete code](./src/part1.py)


---

## **Part 2: Early Stopping**

In this part, we improve our training process by using **early stopping**. Early stopping is a technique that monitors model performance on validation data during training. If the model stops improving for several epochs, training will automatically stop. This helps prevent **overfitting**, where the model performs well on training data but poorly on unseen data.


### **1. Import Required Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
```

**Explanation:**

- `EarlyStopping`: A Keras callback that stops training when performance stops improving.
- Other imports are the same as in Part 1 and support data processing, visualization, and model building.


### **2. Load and Preprocess the Dataset**

```python
cereal_data = pd.read_csv("https://github.com/ML-Course-2025/session5/raw/refs/heads/main/material/datasets/cereal.csv")

cereal_features = cereal_data[['calories', 'protein', 'fat', 'sodium', 'fiber', 'carbo', 'sugars', 'potass', 'vitamins']]
cereal_target = cereal_data['rating']

cereal_X_train, cereal_X_test, cereal_y_train, cereal_y_test = train_test_split(
    cereal_features, cereal_target, test_size=0.2, random_state=42)
```

**Explanation:**

- The cereal dataset is loaded and split into features (inputs) and target (output).
- The data is then divided into training and testing sets using an 80/20 split.


### **3. Define the Neural Network Model**

```python
model = Sequential()
model.add(Input(shape=(cereal_X_train.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))
```

**Explanation:**

- We use a **Sequential** model with three layers.
- The first layer receives the input shape.
- The second and third layers are hidden layers using ReLU activation.
- The final layer outputs a single value, suitable for a regression task.


### **4. Compile the Model**

```python
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
```

**Explanation:**

- The model is compiled using the **Adam** optimizer.
- The **mean squared error** is used as the loss function, which is appropriate for regression.
- The **mean absolute error (MAE)** is used as an evaluation metric.


### **5. Define the Early Stopping Callback**

```python
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)
```

**Explanation:**

This callback automatically stops training under certain conditions:

- `monitor='val_loss'`: Tracks validation loss (performance on the test set).
- `patience=3`: If validation loss does not improve for 3 consecutive epochs, training stops.
- `verbose=1`: Displays a message when early stopping is triggered.
- `restore_best_weights=True`: After stopping, the model’s weights are restored to the best point (lowest validation loss).

This helps prevent the model from learning too much from the training data and overfitting.


### **6. Train the Model with Early Stopping**

```python
history_cereal = model.fit(
    cereal_X_train, cereal_y_train,
    epochs=100,
    batch_size=64,
    validation_data=(cereal_X_test, cereal_y_test),
    callbacks=[early_stopping],
    verbose=1
)
```

**Explanation:**

- We train for up to 100 epochs.
- With early stopping, the model may stop earlier if validation loss doesn't improve.
- `callbacks=[early_stopping]` activates early stopping during training.


### **7. Visualize the Training History**

```python
plt.plot(history_cereal.history['loss'], label='Cereal Train Loss')
plt.plot(history_cereal.history['val_loss'], label='Cereal Validation Loss')
plt.title('Training and Validation Loss for Cereal Dataset')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

**Explanation:**

- This graph shows the training and validation loss over each epoch.
- If the lines start to diverge (training loss goes down while validation loss goes up), that can indicate overfitting.
- Early stopping helps stop training before that happens.



### **Does Early Stopping Help Prevent Overfitting?**

Yes. Here’s how early stopping improves model training:

1. **Prevents Overfitting**: Stops training when the model stops improving on validation data.
2. **Improves Generalization**: Helps the model perform better on unseen data by not overfitting to training data.
3. **Saves Time**: Stops unnecessary epochs and reduces training time.

Early stopping is a practical and efficient way to control training and improve model reliability.

> [!TIP]
> Here's the [complete code](./src/part2.py)

