import pandas as pd
import numpy as np

from pathlib import Path
from scipy import signal
from keras.utils import to_categorical # type: ignore

class EEG:
    def __init__(self, data_dir: Path, intencities: list[float], n_channels: int, image_name: str = "Figs for spectra", use_one_hot: bool = False):
        self.data_dir = data_dir
        self.intencities = intencities
        self.n_channels = n_channels
        self.image_name = image_name
        self.use_one_hot = use_one_hot

        self.ids = None
        self.X = None
        self.y = None

    def load_participant(self, participant_id: int):
        self.int = {}
        self.raw = {}
        self.delta = {}
        self.theta = {}
        self.alpha = {}
        self.beta = {}
        self.gamma = {}

        for i in self.intencities:
            file_name = f"Backgr_int_{i}.dat" if self.image_name == "Figs for spectra" else f"Backgr_int_{i}_type_0.4.dat"
            self.int[i] = np.loadtxt(self.data_dir / f"Participant {participant_id}" / self.image_name / file_name)
            raw_signal = np.empty(self.int[i].shape)
            delta_signal = np.empty(self.int[i].shape)
            theta_signal =  np.empty(self.int[i].shape)
            alpha_signal = np.empty(self.int[i].shape)
            beta_signal = np.empty(self.int[i].shape)
            gamma_signal = np.empty(self.int[i].shape)
            for c in range(self.int[i].shape[1]):
                raw_signal[:, c], delta_signal[:, c], theta_signal[:, c], alpha_signal[:, c], beta_signal[:, c], gamma_signal[:, c] = self._fir_filtering(i, c)

            self.raw[i] = raw_signal
            self.delta[i] = delta_signal
            self.theta[i] = theta_signal
            self.alpha[i] = alpha_signal
            self.beta[i] = beta_signal 
            self.gamma[i] = gamma_signal

        participant_info = {}
        participant_info["raw"] = self.raw
        participant_info["delta"] = self.delta
        participant_info["theta"] = self.theta
        participant_info["alpha"] = self.alpha
        participant_info["beta"] = self.beta
        participant_info["gamma"] = self.gamma

        return participant_info

    def _fir_filtering(self, i, c):
        filter_delta = signal.firwin(400, [1.0, 4.0], pass_zero=False, fs=250)
        filter_theta = signal.firwin(400, [5.0, 8.0], pass_zero=False, fs=250)
        filter_alpha = signal.firwin(400, [8.0, 12.0], pass_zero=False, fs=250)
        filter_beta = signal.firwin(400, [13.0, 30.0], pass_zero=False, fs=250)
        filter_gamma = signal.firwin(400, [31.0, 45.0], pass_zero=False, fs=250)

        res_raw = self.int[i][:, c]
        res_delta = signal.convolve(self.int[i][:, c], filter_delta, mode='same')
        res_theta = signal.convolve(self.int[i][:, c], filter_theta, mode='same')
        res_alpha = signal.convolve(self.int[i][:, c], filter_alpha, mode='same')
        res_beta = signal.convolve(self.int[i][:, c], filter_beta, mode='same')
        res_gamma = signal.convolve(self.int[i][:, c], filter_gamma, mode='same')

        return res_raw, res_delta, res_theta, res_alpha, res_beta, res_gamma

    def _process_signal_type_participant(self, user: int, signal_type: str):
        eeg_person = self.load_participant(user)

        person_signals = []
        signal_index = []
        for i in self.intencities:
            for ch in range(self.n_channels):
                signal_series = eeg_person[signal_type].get(i)[:, ch][:15000]
                person_signals.append(signal_series)

                signal_index.append([user, i, ch])            

        person_signal_index_df = pd.DataFrame(signal_index, columns=["user", "intensity", "channel"])
        person_signal_index_df["is_left_channel"] = np.where(person_signal_index_df["channel"] % 2 == 0, 1, 0)
        person_signal_index_df["signal_type"] = signal_type

        person_signals_df = pd.DataFrame(person_signals)

        return pd.concat([person_signal_index_df, person_signals_df], axis=1)
    
    def load_to_dataframe(self, users: list[int], signals: list[str]):
        data = []
        for user in users:
            for signal_type in signals:
                data.append(self._process_signal_type_participant(user, signal_type))
        data = pd.concat(data, axis=0).reset_index(drop=True)

        self.ids = data.iloc[:, :5]
        self.X = data.iloc[:, 5:].to_numpy()
        if self.use_one_hot:
            self.y = to_categorical((data["intensity"].to_numpy() * 10).astype("int"))[:, 1:]
        else:
            self.y = data["intensity"].to_numpy()
