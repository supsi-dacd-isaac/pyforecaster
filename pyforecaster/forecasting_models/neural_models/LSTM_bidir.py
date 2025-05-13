import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Concatenate, Add
from tensorflow.keras.optimizers import Adam
from pyforecaster.forecaster import ScenarioGenerator



class LSTM_Forecaster(ScenarioGenerator):
    def __init__(self, target_name, lstm_history, pred_horizon=1, q_vect=None, nodes_at_step=None, val_ratio=None, logger=None, n_scen_fit=100,
                 additional_node=False, formatter=None, conditional_to_hour=True, **scengen_kwgs):
        super().__init__(q_vect, val_ratio=val_ratio, nodes_at_step=nodes_at_step, **scengen_kwgs)
        self.target_name = target_name
        self.lstm_history = lstm_history
        self.pred_horizon = pred_horizon
        self.model = None
        self.target_scaler = StandardScaler()
        self.exog_scaler = StandardScaler()

    def _build_model(self, exog_input_shape):
        seq_input = Input(shape=(self.lstm_history, 1), name='lstm_input')
        lstm_out = Bidirectional(LSTM(units=64, return_sequences=False))(seq_input)

        # FFNN for exogenous
        exog_input = Input(shape=(exog_input_shape,), name='exog_input')
        exog_out = Dense(64, activation='relu')(exog_input)
        exog_out = Dense(32, activation='relu')(exog_out)

        # Concatenate LSTM + FF paths
        merged = Concatenate()([lstm_out, exog_out])

        # ⏩ Skip connection from LSTM output to final layer
        lstm_predictor = Dense(self.pred_horizon, activation='linear')(lstm_out)
        exog_predictor = Dense(self.pred_horizon, activation='linear')(exog_out)

        # Add skip and main output
        output = Add()([lstm_predictor, exog_predictor])

        model = Model(inputs=[seq_input, exog_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001),
                      loss='mse',
                      metrics=['mse'])

        return model

    def fit(self, df, y=None, epochs=10, batch_size=32, val_split=0.2, patience=5, **kwargs):
        assert len(df) >= self.lstm_history + self.pred_horizon, "Dataframe too short"

        target_series = df[self.target_name].values.reshape(-1, 1)
        exog_df = df.drop(columns=[self.target_name])

        # Fit scalers
        self.target_scaler.fit(target_series)
        self.exog_scaler.fit(exog_df.values)

        # Transform inputs
        scaled_target = self.target_scaler.transform(target_series).flatten()
        scaled_exog = self.exog_scaler.transform(exog_df.values)

        total_samples = len(df) - self.lstm_history - self.pred_horizon + 1

        X_seq = np.lib.stride_tricks.sliding_window_view(scaled_target, window_shape=self.lstm_history)[:total_samples]
        X_seq = np.expand_dims(X_seq, axis=2)

        X_exog = scaled_exog[self.lstm_history - 1: len(df) - self.pred_horizon]

        y = np.array([
            scaled_target[i + self.lstm_history: i + self.lstm_history + self.pred_horizon]
            for i in range(total_samples)
        ])

        # Time-based split
        split_index = int((1 - val_split) * total_samples)
        X_seq_train, X_seq_val = X_seq[:split_index], X_seq[split_index:]
        X_exog_train, X_exog_val = X_exog[:split_index], X_exog[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        if self.model is None:
            self.model = self._build_model(X_exog.shape[1])

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        self.model.fit(
            [X_seq_train, X_exog_train],
            y_train,
            validation_data=([X_seq_val, X_exog_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            **kwargs
        )

    def predict(self, df, **kwargs):
        assert len(df) >= self.lstm_history, "Dataframe too short for LSTM history"

        target_series = df[self.target_name].values.reshape(-1, 1)
        exog_df = df.drop(columns=[self.target_name])

        scaled_target = self.target_scaler.transform(target_series).flatten()
        scaled_exog = self.exog_scaler.transform(exog_df.values)

        total_samples = len(df) - self.lstm_history

        X_seq = np.lib.stride_tricks.sliding_window_view(scaled_target, window_shape=self.lstm_history)
        X_seq = np.expand_dims(X_seq[:total_samples], axis=2)

        X_exog = scaled_exog[self.lstm_history - 1:]

        preds_scaled = self.model.predict([X_seq, X_exog[:total_samples]])
        preds = self.target_scaler.inverse_transform(preds_scaled)

        # encapsulate predictions in a DataFrame with the same index as the input DataFrame
        columns = np.array([self.target_name + f'_{i+1}' for i in range(self.pred_horizon)])
        preds = pd.DataFrame(preds, index=df.index[self.lstm_history:], columns=columns)


        return preds