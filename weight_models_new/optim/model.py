import tensorflow as tf


class WeightModel:

    def __init__(self, X, Y, pitch_window, history, features):
        """
        Create a new weight model.
        
        Parameters
        ==========
        X : np.ndarray
            N x (history * len(pitches) + num_features + 1) array, where N is the number of data points.
            Each data point contains the acoustic history (as a len(pitches) x history np.ndarray reshaped
            into 1-Dimension, followed by features (if features is True), followed by the language model's
            prior for the corresponding frame.
            
        Y : np.ndarray
            N-length array, containing the target for each data point.
            
        pitch_window : list(int)
            The pitches saved around each data point.

        history : int
            The history length saved in each data point.

        features : boolean
            Whether features are included in the data points (True) or not (False).
        """
        self.X = X
        self.Y = Y
        
        self.pitch_window = pitch_window
        self.history = history
        self.features = features
        self.num_features = X.shape[1] - len(pitch_window) * history - 1
        
        self._prediction = None
        self._optimize = None
        self._error = None
        
        
        
    @property
    def prediction(self):
        if not self._prediction:
            # Split the data into its parts
            acoustic = self.X[:, :len(self.pitch_window) * self.history].reshape(-1, len(self.pitch_window), history)
            features = self.X[:, len(self.pitch_window) * self.history : len(self.pitch_window) * self.history + num_features]
            
            # Run the cnn on the acoustic input
            
            
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
            incoming = tf.matmul(self.data, weight) + bias
            self._prediction = tf.nn.softmax(incoming)
            
        return self._prediction

    
    @property
    def optimize(self):
        if not self._optimize:
            cross_entropy = -tf.reduce_sum(self.target, tf.log(self.prediction))
            optimizer = tf.train.RMSPropOptimizer(0.03)
            self._optimize = optimizer.minimize(cross_entropy)
            
        return self._optimize

    
    @property
    def error(self):
        if not self._error:
            mistakes = tf.not_equal(
                tf.argmax(self.target, 1), tf.argmax(self.prediction, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
            
        return self._error
    