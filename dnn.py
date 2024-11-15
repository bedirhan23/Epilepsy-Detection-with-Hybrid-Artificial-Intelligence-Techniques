import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical

def dnn_model(data):
    X = data.drop(columns=['label'])
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test) 

    model = keras.Sequential([
        keras.layers.Dense(1025, input_shape=(X_train.shape[1],), activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(5, activation='softmax') 
    ])

    model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy}")

if __name__ == "__main__":
    data = pd.read_csv("df_all_cases.csv")
    dnn_model(data)
