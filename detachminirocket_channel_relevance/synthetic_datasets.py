import numpy as np
import matplotlib.pyplot as plt

class SimpleSyntheticDataset():
    def __init__(self, num_samples_train=100, num_samples_test=100, class_balance=0.5,
                 num_channels=5, seq_length=50, important_channels=[2],
                 importance='pulse', A=2, pos=20, length=10, f=10, noise_std=1,
                 random_state=42):
        self.num_samples_train = num_samples_train
        self.num_samples_test = num_samples_test
        self.class_balance = class_balance
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.important_channels = important_channels
        self.importance = importance
        self.f = f
        self.A = A
        self.pos = pos
        self.length = length
        self.noise_std = noise_std
        self.random_state = random_state

        # Set random seed for reproducibility
        np.random.seed(self.random_state)

        # Generate synthetic train and test datasets
        self.train_data, self.train_labels = self.generate_dataset(self.num_samples_train)
        self.test_data, self.test_labels = self.generate_dataset(self.num_samples_test)

    def generate_dataset(self, num_samples):
        time = np.arange(0, 1, 1/self.seq_length)
        class_0_samples = int(num_samples * self.class_balance)
        X = np.empty((num_samples, self.num_channels, self.seq_length))
        y = np.empty((num_samples), dtype=int)

        if self.importance == 'pulse':
            wave = np.zeros(self.seq_length)
            wave[self.pos:self.pos + self.length] = self.A

        elif self.importance == 'sine':
            wave = self.A * np.sin(2 * np.pi * self.f * time)

        elif self.importance == 'absine':
            wave = np.abs(self.A * np.sin(2 * np.pi * self.f * time))

        else: raise ValueError(f'"{self.importance}" importance is not implemented. Choose one from "pulse" / "sine" / "absine".')

        for sample in range(class_0_samples):
            for channel in range(self.num_channels):
                X[sample, channel] = np.random.normal(0, self.noise_std, self.seq_length)
                if channel in self.important_channels: X[sample, channel] += wave

            y[sample] = 0

        for sample in range(class_0_samples, num_samples):
            for channel in range(self.num_channels):
                X[sample, channel] = np.random.normal(0, self.noise_std, self.seq_length)

            y[sample] = 1

        return X, y

    def plot(self, sample0=0, sample1=-1):
        fig, axes = plt.subplots(1, self.num_channels, figsize=(3*self.num_channels, 4))
        fig.suptitle('Informative class sample', fontsize=14)

        for ch, ax in enumerate(axes):
            if ch in self.important_channels: ax.plot(self.train_data[sample0, ch], color='orange')
            else: ax.plot(self.train_data[sample0, ch])
            ax.set_title(f'Channel {ch + 1}')
            ax.set_xlabel('Time steps')
            if ch == 0: ax.set_ylabel('Amplitude')
            ax.set_ylim(-4, 4)


        fig, axes = plt.subplots(1, self.num_channels, figsize=(3*self.num_channels, 4))
        fig.suptitle('Non-informative class sample', fontsize=14)

        for ch, ax in enumerate(axes):
            ax.plot(self.train_data[sample1, ch])
            ax.set_title(f'Channel {ch + 1}')
            ax.set_xlabel('Time steps')
            if ch == 0: ax.set_ylabel('Amplitude')
            ax.set_ylim(-4, 4)