import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, BatchNormalization, MaxPooling1D, Flatten, Dense, LeakyReLU, ReLU, Dropout, Lambda, Concatenate
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal
from sklearn.model_selection import KFold
import optuna
import os
import json
from datetime import datetime

# ------------------------------------ Read Data ------------------------------------
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

# ------------------------------------ Convert to Numpy Array ------------------------
train["Seq1"] = train["Seq1"].apply(lambda x: np.fromstring(x, sep=','))
train["Seq2"] = train["Seq2"].apply(lambda x: np.fromstring(x, sep=','))

test["Seq1"] = test["Seq1"].apply(lambda x: np.fromstring(x, sep=','))
test["Seq2"] = test["Seq2"].apply(lambda x: np.fromstring(x, sep=','))

# ------------------------------------ Normalization ---------------------------------
train["Seq1"] = train["Seq1"].apply(lambda x: x / 4.0)
train["Seq2"] = train["Seq2"].apply(lambda x: x / 4.0)
test["Seq1"] = test["Seq1"].apply(lambda x: x / 4.0)
test["Seq2"] = test["Seq2"].apply(lambda x: x / 4.0)

# ------------------------------------ Model Functions --------------------------------
def create_base_model(input_shape):
    input_layer = Input(shape=input_shape)
    layer = Conv1D(16, 5, kernel_initializer=HeNormal(), activation=None)(input_layer)
    layer = BatchNormalization(momentum=0.9)(layer)
    layer = LeakyReLU(alpha=0.1)(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Dropout(0.3)(layer)
    layer = Conv1D(32, 5, kernel_initializer=HeNormal(), activation=None)(input_layer)
    layer = BatchNormalization(momentum=0.75)(layer)
    layer = LeakyReLU(alpha=0.1)(layer)
    layer = MaxPooling1D(2)(layer)
    layer = Dropout(0.5)(layer)
    layer = Flatten()(layer)
    layer = Dense(64, kernel_initializer=HeNormal(), activation=None)(layer)
    layer = BatchNormalization(momentum=0.65)(layer)
    layer = LeakyReLU(alpha=0.1)(layer)
    return Model(inputs=input_layer, outputs=layer)

def Siamese(input_shape, learning_rate, dropout_rates, dense_units):
    base_model = create_base_model(input_shape)

    input_1 = Input(shape=input_shape)
    input_2 = Input(shape=input_shape)

    branch_1 = base_model(input_1)
    branch_2 = base_model(input_2)

    merged = Concatenate()([branch_1, branch_2])
    dense = Dense(dense_units, activation="relu")(merged)
    dense = Dropout(dropout_rates[2])(dense)
    output = Dense(1, activation="linear")(dense)

    siamese_model = Model(inputs=[input_1, input_2], outputs=output)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                         loss="mse", 
                         metrics=["mae", "mse"])
    return siamese_model

# ------------------------------------ Data Preparation -----------------------------
train_seq1 = np.array((train["Seq1"]).tolist())
train_seq2 = np.array((train["Seq2"]).tolist())
train_corr = np.array(train["Corr"].tolist())

test_seq1 = np.array((test["Seq1"]).tolist())
test_seq2 = np.array((test["Seq2"]).tolist())
test_corr = np.array(test["Corr"].tolist())

X_train = list(zip(train_seq1, train_seq2))
y_train = train["Corr"].values

X_test = list(zip(test_seq1, test_seq2))
y_test = test["Corr"].values

INPUT_SHAPE = (2000, 1)

# -------------------------- Optuna for Hyperparameter Tuning -------------------------
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout_rates = [trial.suggest_float(f"dropout_{i}", 0.1, 0.5) for i in range(3)]
    dense_units = trial.suggest_int("dense_units", 16, 64, step=16)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 20, 50)

    k = 5
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    val_maes = []

    for fold, (train_index, val_index) in enumerate(kf.split(X_train)):
        X_train_fold = [X_train[i] for i in train_index]
        y_train_fold = y_train[train_index]
        X_val_fold = [X_train[i] for i in val_index]
        y_val_fold = y_train[val_index]

        X_train_fold_seq1 = np.array([x[0] for x in X_train_fold])
        X_train_fold_seq2 = np.array([x[1] for x in X_train_fold])
        X_val_fold_seq1 = np.array([x[0] for x in X_val_fold])
        X_val_fold_seq2 = np.array([x[1] for x in X_val_fold])

        siamese_model = Siamese(INPUT_SHAPE, learning_rate, dropout_rates, dense_units)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = siamese_model.fit(
            [X_train_fold_seq1, X_train_fold_seq2], y_train_fold,
            validation_data=([X_val_fold_seq1, X_val_fold_seq2], y_val_fold),
            epochs=epochs, batch_size=batch_size, callbacks=[lr_scheduler, early_stopping],
            verbose=0
        )

        val_loss, val_mae, _ = siamese_model.evaluate([X_val_fold_seq1, X_val_fold_seq2], y_val_fold, verbose=0)
        val_maes.append(val_mae)

    mean_val_mae = np.mean(val_maes)
    return mean_val_mae

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("Best parameters:", best_params)

# ------------------------------------ Create Results Directory --------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = f"results_{timestamp}"
os.makedirs(results_dir, exist_ok=True)

# ------------------------------------ Save Best Parameters ------------------------------------
params_path = os.path.join(results_dir, "best_params.json")
with open(params_path, 'w') as f:
    json.dump(best_params, f, indent=4)
print(f"Saved best parameters to {params_path}")

# ------------------------------------ Train Final Model with Best Parameters ------------------
best_learning_rate = best_params["learning_rate"]
best_dropout_rates = [best_params[f"dropout_{i}"] for i in range(3)]
best_dense_units = best_params["dense_units"]
best_batch_size = best_params["batch_size"]
best_epochs = best_params["epochs"]

final_model = Siamese(INPUT_SHAPE, best_learning_rate, best_dropout_rates, best_dense_units)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = final_model.fit(
    [train_seq1, train_seq2], train_corr,
    validation_split=0.2,
    epochs=best_epochs, batch_size=best_batch_size, callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# ------------------------------------ Plot Training and Validation Loss ----------------------
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
plt.title('Training and Validation Loss Over Epochs', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
loss_plot_path = os.path.join(results_dir, "loss_plot.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print(f"Saved loss plot to {loss_plot_path}")

# ------------------------------------ Evaluate and Save Results -------------------------------
train_preds = final_model.predict([train_seq1, train_seq2])
test_preds = final_model.predict([test_seq1, test_seq2])

train_mse = mean_squared_error(train_corr, train_preds)
train_mae = mean_absolute_error(train_corr, train_preds)
test_mse = mean_squared_error(test_corr, test_preds)
test_mae = mean_absolute_error(test_corr, test_preds)

results = pd.DataFrame({
    "Metric": ["Train_MSE", "Train_MAE", "Test_MSE", "Test_MAE"],
    "Value": [train_mse, train_mae, test_mse, test_mae]
})
results.to_csv(os.path.join(results_dir, "metrics.csv"), index=False)

train_predictions_df = pd.DataFrame({"True": train_corr, "Predicted": train_preds.flatten()})
test_predictions_df = pd.DataFrame({"True": test_corr, "Predicted": test_preds.flatten()})
train_predictions_df.to_csv(os.path.join(results_dir, "train_predictions.csv"), index=False)
test_predictions_df.to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)

# Save model's weights
final_model.save_weights(os.path.join(results_dir, "model.weights.h5"))

print(f"Results saved in directory: {results_dir}")