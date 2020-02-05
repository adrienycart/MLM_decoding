from mlm_training.dataset import Dataset, ground_truth
from mlm_training.utils import safe_mkdir
import os
import tensorflow as tf
import numpy as np
from datetime import datetime

"""
Note
----

All the nodes of the graph are implemented as properties.
That way, when the attribute is called for the first time, the corresponding
graph nodes are created. The following times, the attribute is just returned,
without adding new nodes to the graph.
"""

class Model:

    def __init__(self, model_param):
        tf.reset_default_graph()

        #Unpack parameters
        for key,value in model_param.items():
            print(key,value)
            setattr(self,key,value)

        if self.pitchwise:
            self.n_classes = 1
        else:
            self.n_classes = self.n_notes

        self._batch_size = None
        self._inputs = None
        self._acoustic_outputs = None
        self._seq_lens = None
        self._labels = None
        self._key_masks = None
        self._thresh = None
        self._thresh_key = None
        self._thresh_active = None
        self._sched_samp_p = None
        self._gamma = None

        self.initial_state = None
        self.output_state = None

        self._prediction = None
        self._pred_sigm = None
        self._last_pred_sigm = None
        self._prediction_sched_samp = None
        self._pred_thresh = None
        self._cross_entropy = None
        self._cross_entropy2 = None
        self._cross_entropy_transition = None
        self._cross_entropy_transition2 = None
        self._cross_entropy_length = None
        self._cross_entropy_length2 = None
        self._cross_entropy_steady = None
        self._cross_entropy_steady2 = None
        self._cross_entropy_key = None
        self._cross_entropy_key2 = None

        self._combined_metric = None
        self._combined_metric2 = None
        self._combined_metric_norm = None
        self._combined_metric_norm2 = None

        self._focal_loss = None
        self._loss = None
        self._optimize = None
        self._tp = None
        self._fp = None
        self._fn = None
        self._precision = None
        self._recall = None
        self._f_measure = None


        #Call to create the graph
        self.build_graph()


    def _transpose_data(self, data):
        return np.transpose(data,[0,2,1])

    def print_params(self):
        print("Learning rate : ",self.learning_rate)
        print("Hidden nodes : ",self.n_hidden)
        if not type(self.n_hidden)==int:
            print("Activation function : ",self.activ)
        if self.chunks:
            print("Chunks : ",self.chunks)


    @property
    def tp(self):
        """
        Number of true positives per sequence
        """
        if self._tp is None:
            with tf.device(self.device_name):
                pred = self.pred_thresh

                y = self.labels

                if self.with_onsets:
                    bool_matrix = tf.logical_and(tf.greater_equal(pred,1),tf.greater_equal(y,1))
                else:
                    bool_matrix = tf.logical_and(tf.equal(pred,1),tf.equal(y,1))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._tp = reduced
        return self._tp

    @property
    def fp(self):
        """
        Number of false positives per sequence
        """
        if self._fp is None:
            with tf.device(self.device_name):
                pred = self.pred_thresh

                y = self.labels
                if self.with_onsets:
                    bool_matrix = tf.logical_and(tf.greater_equal(pred,1),tf.equal(y,0))
                else:
                    bool_matrix = tf.logical_and(tf.equal(pred,1),tf.equal(y,0))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._fp = reduced
        return self._fp

    @property
    def fn(self):
        """
        Number of false negatives per sequence
        """
        if self._fn is None:
            with tf.device(self.device_name):
                pred = self.pred_thresh

                y = self.labels
                if self.with_onsets:
                    bool_matrix = tf.logical_and(tf.equal(pred,0),tf.greater_equal(y,0))
                else:
                    bool_matrix = tf.logical_and(tf.equal(pred,0),tf.equal(y,1))
                reduced = tf.reduce_sum(tf.cast(bool_matrix,tf.float32),[1,2])
                self._fn = reduced
        return self._fn


    @property
    def precision(self):
        """
        Precision per sequence.
        Returns a vector of length len(dataset), mean has to be computed afterwards
        """

        if self._precision is None:
            with tf.device(self.device_name):
                TP = self.tp
                FP = self.fp
                self._precision = tf.truediv(TP,tf.add(tf.add(TP,FP),1e-6))
        return self._precision


    @property
    def recall(self):
        """
        Recall per sequence.
        Returns a vector of length len(dataset), mean has to be computed afterwards
        """
        if self._recall is None:
            with tf.device(self.device_name):
                TP = self.tp
                FN = self.fn
                self._recall = tf.truediv(TP,tf.add(tf.add(TP,FN),1e-6))
        return self._recall

    @property
    def f_measure(self):
        """
        F-measure per sequence.
        Returns a vector of length len(dataset), mean has to be computed afterwards
        """
        if self._f_measure is None:
            with tf.device(self.device_name):
                prec = self.precision
                rec = self.recall
                self._f_measure = tf.truediv(tf.scalar_mul(2,tf.multiply(prec,rec)),tf.add(tf.add(prec,rec),1e-6))
        return self._f_measure

    @property
    def batch_size(self):
        """
        Placeholder for batch size
        """
        if self._batch_size is None:
            suffix = self.suffix
            batch_size = tf.placeholder("int32",(),name="batch_size"+suffix)
            self._batch_size = batch_size
        return self._batch_size


    @property
    def inputs(self):
        """
        Placeholder for input sequences
        """
        if self._inputs is None:
            n_notes = self.n_notes
            n_steps = self.n_steps
            suffix = self.suffix

            x = tf.placeholder("float", [None,n_steps,n_notes],name="x"+suffix)

            self._inputs = x
        return self._inputs

    @property
    def acoustic_outputs(self):
        """
        Placeholder for input sequences
        """
        if self._acoustic_outputs is None:
            n_notes = self.n_notes
            n_steps = self.n_steps
            suffix = self.suffix

            ac = tf.placeholder("float", [None,n_steps,n_notes],name="acoustic_outputs"+suffix)

            self._acoustic_outputs = ac
        return self._acoustic_outputs

    @property
    def seq_lens(self):
        """
        Placeholder for sequence lengths (not used at the moment)
        """
        if self._seq_lens is None:
            suffix = self.suffix
            seq_len = tf.placeholder("int32",[None], name="seq_len"+suffix)

            self._seq_lens = seq_len
        return self._seq_lens


    @property
    def sched_samp_p(self):
        """
        Placeholder for scheduled sampling probability
        """
        if self._sched_samp_p is None:
            suffix = self.suffix
            sched_samp_p = tf.placeholder("float",shape=[],name="sched_samp_p"+suffix)
            self._sched_samp_p = sched_samp_p
        return self._sched_samp_p

    # @property
    # def inputs_sched_samp(self):
    #     """
    #     Remplace some inputs with sampled values
    #     """
    #     if self._inputs_sched_samp is None:
    #
    #         prediction_estimates = tf.stop_gradient(self.pred_sigm)
    #         sched_samp_p = self.sched_samp_p
    #
    #         #pred is : Batch size, n_steps, n_notes
    #         distrib_indices = tf.distributions.Bernoulli(probs=sched_samp_p,type=tf.bool)
    #         sample_indices = distrib_indices.sample(shape=tf.get_shape(tf.prediction_estimates)[0:2])
    #
    #         frames_to_sample =  prediction_estimates[sample_indices]
    #         distrib_frames = tf.distributions.Bernouilli(probs=frames_to_sample,type=tf.float32)
    #         sampled_frames = distrib_frames.sample()
    #
    #         input_sampled = self.inputs
    #         input_sampled[sample_indices] = sampled_frames
    #
    #
    #         self._inputs_sched_samp = input_sampled
    #     return self._inputs_sched_samp



    @property
    def prediction(self):
        """
        Logit predictions: x[t] given x[0:t]
        """
        if self._prediction is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_classes = self.n_classes
                n_steps = self.n_steps
                n_hidden = self.n_hidden
                suffix = self.suffix
                batch_size = self.batch_size

                x = self.inputs
                seq_len = self.seq_lens

                if self.cell_type == "LSTM":
                    cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True,forget_bias = 1.0)
                elif self.cell_type == "diagLSTM":
                    cell = ModLSTMCell(n_hidden,tf.truncated_normal_initializer())
                else:
                    raise ValueError("model_param['cell_type'] not understood!")


                if self.with_onsets:

                    n_outputs = 3
                    x_expanded = tf.one_hot(tf.cast(x,tf.int32),depth=n_outputs,dtype=tf.float32)
                    x_flat = tf.reshape(x_expanded,[-1,n_steps,n_classes*n_outputs])

                    W = tf.Variable(tf.truncated_normal([n_hidden,n_classes*n_outputs]),name="W"+suffix)
                    b = tf.Variable(tf.truncated_normal([n_classes*n_outputs]),name="b"+suffix)

                    hidden_state_in = cell.zero_state(batch_size, dtype=tf.float32)
                    self.initial_state = hidden_state_in

                    #We don't take into account sequence length because doing so
                    #causes Tensorflow to output weird results
                    outputs, hidden_state_out = tf.nn.dynamic_rnn(cell,x_flat,initial_state=hidden_state_in,
                        dtype=tf.float32,time_major=False)#,sequence_length=seq_len)
                    self.output_state = hidden_state_out

                    outputs = tf.reshape(outputs,[-1,n_hidden])
                    pred = tf.matmul(outputs,W) + b
                    pred = tf.reshape(pred,[-1,n_steps,n_classes,n_outputs])

                else:
                    W = tf.Variable(tf.truncated_normal([n_hidden,n_classes]),name="W"+suffix)
                    b = tf.Variable(tf.truncated_normal([n_classes]),name="b"+suffix)

                    hidden_state_in = cell.zero_state(batch_size, dtype=tf.float32)
                    self.initial_state = hidden_state_in

                    #We don't take into account sequence length because doing so
                    #causes Tensorflow to output weird results
                    outputs, hidden_state_out = tf.nn.dynamic_rnn(cell,x,initial_state=hidden_state_in,
                        dtype=tf.float32,time_major=False)#,sequence_length=seq_len)
                    self.output_state = hidden_state_out

                    outputs = tf.reshape(outputs,[-1,n_hidden])
                    pred = tf.matmul(outputs,W) + b

                    pred = tf.reshape(pred,[-1,n_steps,n_classes])

                self._prediction = pred
        return self._prediction


    @property
    def prediction_sched_samp(self):
        """
        Logit predictions: x[t] given x[0:t]
        """
        if self._prediction_sched_samp is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_classes = self.n_classes
                n_steps = self.n_steps
                n_hidden = self.n_hidden
                suffix = self.suffix
                batch_size = self.batch_size

                inputs = tf.concat([self.inputs,tf.zeros_like(self.inputs[:,0:1,:])],axis=1)
                if self.with_onsets:
                    n_outputs = 3
                    x_expanded = tf.one_hot(tf.cast(inputs,tf.int32),depth=n_outputs,dtype=tf.float32)
                    # n_steps+1 because we added an extra step of zeros above
                    inputs = tf.reshape(x_expanded,[-1,n_steps+1,n_classes*n_outputs])
                    activation = tf.nn.softmax
                    dense_outputs = n_classes*n_outputs
                else:
                    activation = tf.nn.sigmoid
                    dense_outputs = n_classes

                # Shape = [time,batch,pitch]
                inputs = tf.transpose(inputs,[1,0,2])


                # inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                #                     dtype=tf.float32)
                sequence_length = self.seq_lens
                inputs_ta = tf.TensorArray(dtype=tf.float32, size=n_steps+1)
                inputs_ta = inputs_ta.unstack(inputs)


                if self.cell_type=="LSTM":
                    cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True,forget_bias = 1.0)
                elif self.cell_type=="diagLSTM":
                    cell = ModLSTMCell(n_hidden,tf.truncated_normal_initializer())
                else:
                    raise ValueError("model_param['cell_type'] not understood!")

                hidden_state_in = cell.zero_state(batch_size, dtype=tf.float32)
                self.initial_state = hidden_state_in
                sched_samp_p = self.sched_samp_p

                if self.scheduled_sampling == 'mix':

                    acoustic = tf.concat([self.acoustic_outputs,tf.zeros_like(self.acoustic_outputs[:,0:1,:])],axis=1)
                    acoustic = tf.transpose(acoustic,[1,0,2])
                    # inputs = tf.placeholder(shape=(max_time, batch_size, input_depth),
                    #                     dtype=tf.float32)
                    acoustic_ta = tf.TensorArray(dtype=tf.float32, size=n_steps+1)
                    acoustic_ta = acoustic_ta.unstack(acoustic)


                def loop_fn(time, cell_output, cell_state, loop_state):
                    emit_output = cell_output  # == None for time == 0

                    if cell_output is None:  # time == 0
                        next_cell_state = hidden_state_in
                        next_input = inputs_ta.read(time)
                    else:
                        next_cell_state = cell_state

                        use_real_input = tf.distributions.Bernoulli(probs=sched_samp_p,dtype=tf.bool).sample()

                        if self.scheduled_sampling == 'mix':
                            w = self.sampl_mix_weight
                            prev_probs = tf.nn.sigmoid(tf.layers.dense(cell_output, n_classes,
                                name='output_layer',
                                reuse=tf.AUTO_REUSE))
                            next_input = tf.cond(use_real_input,
                                    lambda: inputs_ta.read(time),
                                    lambda: tf.contrib.distributions.Bernoulli(
                                    # probs=tf.nn.sigmoid(dense_layer(cell_output,W,b,1)),
                                    probs=w*acoustic_ta.read(time)+(1-w)*prev_probs,
                                    dtype=tf.float32).sample())
                        elif self.scheduled_sampling == 'self':
                            next_input = tf.cond(use_real_input,
                                    lambda: inputs_ta.read(time),
                                    lambda: tf.contrib.distributions.Bernoulli(
                                    # probs=tf.nn.sigmoid(dense_layer(cell_output,W,b,1)),
                                    probs=activation(tf.layers.dense(cell_output, dense_outputs,
                                        name='output_layer',
                                        reuse=tf.AUTO_REUSE)),
                                    dtype=tf.float32).sample())

                    next_loop_state = None


                    elements_finished = (time >= n_steps)

                    return (elements_finished, next_input, next_cell_state,
                          emit_output, next_loop_state)

                outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
                outputs = outputs_ta.stack()

                self.output_state = final_state

                # pred = dense_layer(outputs)
                pred = tf.layers.dense(outputs, dense_outputs,name='rnn/output_layer',reuse=tf.AUTO_REUSE)
                pred = tf.transpose(pred,[1,0,2])

                if self.with_onsets:
                    pred = tf.reshape(pred,[-1,n_steps,n_classes,n_outputs])

                self._prediction_sched_samp = pred
        return self._prediction_sched_samp





    @property
    def pred_sigm(self):
        """
        Sigmoid predictions: x[t] given x[0:t]
        """
        if self._pred_sigm is None:
            with tf.device(self.device_name):
                if self.scheduled_sampling:
                    pred = self.prediction_sched_samp
                else:
                    pred = self.prediction
                if self.with_onsets :
                    pred = tf.nn.softmax(pred)
                else:
                    pred = tf.sigmoid(pred)
                self._pred_sigm = pred
        return self._pred_sigm


    @property
    def thresh(self):
        """
        Placeholder for threshold (to apply to the sigmoid predictions)
        """
        if self._thresh is None:
            suffix = self.suffix
            thresh = tf.placeholder_with_default(0.5,shape=[],name="thresh"+suffix)
            self._thresh = thresh
        return self._thresh

    @property
    def pred_thresh(self):
        """
        Thresholded predictions, using self.thresh placeholder
        """
        if self._pred_thresh is None:
            with tf.device(self.device_name):
                if self.with_onsets:
                    pred = tf.argmax(self.pred_sigm,axis=3)
                else:
                    thresh = self.thresh

                    pred = self.pred_sigm
                    pred = tf.greater(pred,thresh)
                    pred = tf.cast(pred,tf.int8)
                self._pred_thresh = pred
        return self._pred_thresh



    @property
    def labels(self):
        """
        Placeholder for targets (shifted version of inputs in the case of prediction)
        """
        if self._labels is None:
            n_classes = self.n_classes
            n_steps = self.n_steps
            suffix = self.suffix

            y = tf.placeholder("float", [None,n_steps,n_classes],name="y"+suffix)

            self._labels = y
        return self._labels

    @property
    def key_masks(self):
        if self._key_masks is None:
            suffix = self.suffix
            n_steps = self.n_steps
            n_classes = self.n_classes
            key_masks = tf.placeholder('float',[None,n_steps,n_classes],name="key_masks"+suffix)

            self._key_masks = key_masks
        return self._key_masks

    @property
    def cross_entropy(self):
        """
        Mean cross entropy
        """
        if self._cross_entropy is None:
            with tf.device(self.device_name):
                cross_entropy = self.cross_entropy2
                cross_entropy = tf.reduce_mean(cross_entropy)
                self._cross_entropy = cross_entropy
        return self._cross_entropy


    @property
    def cross_entropy2(self):
        """
        Cross entropy as a vector of length batch_size
        """
        if self._cross_entropy2 is None:
            with tf.device(self.device_name):
                n_steps = self.n_steps
                suffix = self.suffix
                y = self.labels
                if self.scheduled_sampling:
                    pred = self.prediction_sched_samp
                else:
                    pred = self.prediction

                if self.with_onsets:
                    cross_entropy2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=tf.cast(y,tf.int32)),axis=[1,2])
                else:
                    cross_entropy2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y),axis=[1,2])
                self._cross_entropy2 = cross_entropy2
        return self._cross_entropy2

    def split_steady(self, x,*args):
        #x is one single input, same for all tensors in args
        data_extended = tf.pad(x,[[1,1],[0,0]],'constant')
        diff = data_extended[1:,:] - data_extended[:-1,:]
        # diff = tf.split(diff,tf.shape(diff)[0])
        trans_mask = tf.logical_or(tf.equal(diff,1), tf.equal(diff,-1))
        steady = tf.where(tf.logical_not(trans_mask))
        # lol = transitions[:,1,:]

        steady_unique, _ ,count_steady = tf.unique_with_counts(steady[:,0])
        steady_unique = tf.where(tf.equal(count_steady,self.n_notes))[:,0]
        steady_unique_steps = tf.gather(steady_unique,tf.where(tf.logical_and(tf.logical_not(tf.equal(steady_unique,0)),tf.logical_not(tf.equal(steady_unique,tf.cast(tf.shape(x)[0],tf.int64)))))[:,0])
        # steady_unique = steady_unique[:-1]
        # steady_unique_steps = tf.Print(steady_unique_steps,[seq_len,tf.shape(x)[0]])
        out = []
        for tensor in args:
            out += [tf.gather(tensor,tf.add(steady_unique_steps,-1))]
        out += [count_steady]

        return out

    #def split_trans(self, x,y,pred):
    def split_trans(self, x,*args):
        #x is one single input, same for all tensors in args
        data_extended = tf.pad(x,[[1,1],[0,0]],'constant')
        diff = data_extended[1:,:] - data_extended[:-1,:]
        # diff = tf.split(diff,tf.shape(diff)[0])
        trans_mask = tf.logical_or(tf.equal(diff,1), tf.equal(diff,-1))
        transitions= tf.where(trans_mask)

        transitions_unique, _ ,count_trans = tf.unique_with_counts(transitions[:,0])
        #We drop the first onset only if it is 0
        idx_to_keep = tf.where(tf.logical_and(tf.logical_not(tf.equal(transitions_unique,0)),tf.logical_not(tf.equal(transitions_unique,tf.cast(tf.shape(x)[0],tf.int64)))))[:,0]
        transitions_unique_trim = tf.gather(transitions_unique,idx_to_keep)
        count_trans_trim = tf.gather(count_trans,idx_to_keep)
        transitions_unique_trim = transitions_unique_trim
        out = []

        # pred_trans = tf.gather(pred,tf.add(transitions_unique,-1))
        # y_trans = tf.gather(y,tf.add(transitions_unique,-1))
        for tensor in args:
            out += [tf.gather(tensor,tf.add(transitions_unique_trim,-1))]
        out += [count_trans_trim]

        #return pred_trans, y_trans, count_trans
        return out

    @property
    def cross_entropy_transition2(self):
        if self._cross_entropy_transition2 is None:
            with tf.device(self.device_name):

                def compute_one(elems):
                    x = elems[0]
                    y = elems[1]
                    pred = elems[2]
                    seq_len = elems[3]

                    x = x[:seq_len,:]
                    y = y[:seq_len,:]
                    pred = pred[:seq_len,:]

                    y, pred, count = self.split_trans(x,y,pred)


                    cross_entropy_trans = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
                    cross_entropy_trans = tf.reduce_mean(cross_entropy_trans,axis=1)
                    cross_entropy_trans = tf.reduce_mean(tf.div(cross_entropy_trans,tf.cast(count,tf.float32)))
                    # cross_entropy_trans = tf.Print(cross_entropy_trans,[cross_entropy_trans],message="trans",summarize=1000000)
                    #It is necessary that the output has the same dimensions as input (even if not used)
                    return cross_entropy_trans, tf.cast(tf.shape(pred),tf.float32), 0.0,0

                if hasattr(self, 'scheduled_sampling') and self.scheduled_sampling:
                    pred = self.prediction_sched_samp
                else:
                    pred = self.prediction
                xs = self.inputs
                ys = self.labels
                seq_lens = self.seq_lens


                XEs = tf.map_fn(compute_one,[xs,ys,pred,seq_lens],dtype=(tf.float32,tf.float32,tf.float32,tf.int32))
                cross_entropy_trans = XEs[0]
                # cross_entropy_trans = tf.Print(cross_entropy_trans,[tf.where(tf.is_nan(cross_entropy_trans)),XEs[1]],message="trans",summarize=1000000)
                # pred_test, y_test , count_test = self.split_trans(xs[0],ys[0],pred[0])
                # test1 = tf.identity([pred_test,y_test],name='test1')

                self._cross_entropy_transition2 = cross_entropy_trans
        return self._cross_entropy_transition2

    @property
    def cross_entropy_transition(self):
        if self._cross_entropy_transition is None:
            with tf.device(self.device_name):

                XEs = self.cross_entropy_transition2
                XEs = tf.gather(XEs,tf.where(tf.logical_not(tf.is_nan(XEs))))
                cross_entropy_trans = tf.reduce_mean(XEs)

                self._cross_entropy_transition = cross_entropy_trans
        return self._cross_entropy_transition


    @property
    def cross_entropy_steady2(self):
        if self._cross_entropy_steady2 is None:
            with tf.device(self.device_name):

                def compute_one(elems):
                    x = elems[0]
                    y = elems[1]
                    pred = elems[2]
                    seq_len = elems[3]

                    x = x[:seq_len,:]
                    y = y[:seq_len,:]
                    pred = pred[:seq_len,:]

                    y, pred, _ = self.split_steady(x,y,pred)

                    cross_entropy_steady = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
                    cross_entropy_steady = tf.reduce_mean(cross_entropy_steady)
                    # cross_entropy_steady = tf.Print(tf.reduce_mean(cross_entropy_steady),[tf.shape(cross_entropy_steady),tf.reduce_sum(cross_entropy_steady),tf.reduce_mean(cross_entropy_steady)],message="steady")
                    # output = tf.cond(tf.equal(tf.reduce_sum(cross_entropy_steady),0),
                    #         fn1 = lambda: tf.Print(0.0,[0],message='steady zero'),
                    #         fn2 = lambda: tf.Print(cross_entropy_steady,[tf.shape(cross_entropy_steady),tf.reduce_sum(cross_entropy_steady),tf.reduce_mean(cross_entropy_steady)],message="steady"))
                    # cross_entropy_steady = tf.Print(cross_entropy_steady,[cross_entropy_steady,tf.shape(pred),tf.shape(x),seq_len],message="steady",summarize=1000000)

                    #It is necessary that the output has the same dimensions as input (even if not used)
                    return cross_entropy_steady, tf.cast(tf.shape(pred),tf.float32), 0.0, 0

                if hasattr(self, 'scheduled_sampling') and self.scheduled_sampling:
                    pred = self.prediction_sched_samp
                else:
                    pred = self.prediction
                xs = self.inputs
                ys = self.labels
                seq_lens = self.seq_lens

                XEs = tf.map_fn(compute_one,[xs,ys,pred,seq_lens],dtype=(tf.float32,tf.float32,tf.float32,tf.int32))
                cross_entropy_steady = XEs[0]
                # cross_entropy_steady = tf.Print(cross_entropy_steady,[tf.where(tf.is_nan(cross_entropy_steady)),XEs[1]],message="steady",summarize=1000000)

                self._cross_entropy_steady2 = cross_entropy_steady
        return self._cross_entropy_steady2

    @property
    def cross_entropy_steady(self):
        if self._cross_entropy_steady is None:
            with tf.device(self.device_name):


                XEs = self.cross_entropy_steady2
                XEs_no_nan = tf.gather(XEs,tf.where(tf.logical_not(tf.is_nan(XEs))))
                cross_entropy_steady = tf.reduce_mean(XEs_no_nan)


                self._cross_entropy_steady = cross_entropy_steady
        return self._cross_entropy_steady

    @property
    def cross_entropy_length2(self):
        if self._cross_entropy_length2 is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_steps = self.n_steps
                suffix = self.suffix
                y = self.labels
                seq_len = self.seq_lens

                if hasattr(self, 'scheduled_sampling') and self.scheduled_sampling:
                    pred = self.prediction_sched_samp
                else:
                    pred = self.prediction

                mask = tf.sequence_mask(seq_len,maxlen=n_steps)
                mask = tf.expand_dims(mask,-1)
                mask = tf.tile(mask,[1,1,n_notes])

                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
                cross_entropy_masked = cross_entropy*tf.cast(mask,tf.float32)

                cross_entropy_length = tf.reduce_mean(cross_entropy_masked*n_steps,axis=[1,2])/tf.cast(seq_len,tf.float32)

                self._cross_entropy_length2 = cross_entropy_length
        return self._cross_entropy_length2

    @property
    def cross_entropy_length(self):
        if self._cross_entropy_length is None:
            with tf.device(self.device_name):
                cross_entropy_length = self.cross_entropy_length2
                cross_entropy_length = tf.reduce_mean(cross_entropy_length)

                self._cross_entropy_length = cross_entropy_length
        return self._cross_entropy_length


    @property
    def thresh_key(self):
        if self._thresh_key is None:
            with tf.device(self.device_name):
                thresh_key = tf.placeholder_with_default(0.05, shape=(), name="thresh_key"+self.suffix)
                self._thresh_key = thresh_key
        return self._thresh_key


    @property
    def cross_entropy_key2(self):
        if self._cross_entropy_key2 is None:
            with tf.device(self.device_name):

                def transitions_one(elems):
                    tensor,x,active_mask,key_mask,seq_len = elems

                    x = x[:seq_len,:]
                    tensor = tensor[:seq_len,:]
                    active_mask = active_mask[:seq_len,:]
                    key_mask = key_mask[:seq_len,:]

                    tensor,active_mask,key_mask,_ = self.split_trans(x,tensor,active_mask,key_mask)
                    XE = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor, labels=key_mask)*active_mask
                    XE = tf.reduce_sum(XE)/tf.reduce_sum(active_mask)
                    return XE, 0.0,0.0,0.0,0

                def transitions_one_norm(elems):
                    tensor,x,seq_len,active_mask,key_mask  = elems

                    x = x[:seq_len,:]
                    tensor = tensor[:seq_len,:]
                    active_mask = active_mask[:seq_len,:]
                    key_mask = key_mask[:seq_len,:]

                    tensor_split,active_mask_split,key_mask_split,_ = self.split_trans(x,tensor,active_mask,key_mask)
                    XE = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor_split, labels=key_mask_split)*active_mask_split
                    norm_factor = tf.expand_dims(tf.reduce_sum(key_mask_split,axis=[1]),axis=1)
                    XE = XE/norm_factor
                    # XE = tf.where(tf.is_nan(XE), tf.zeros_like(XE), XE)
                    XE= tf.reduce_sum(XE)/tf.reduce_sum(active_mask_split)
                    # XE = tf.Print(XE,[tf.where(tf.equal(norm_factor,0)),tf.where(tf.equal(tf.reduce_sum(key_mask_split,axis=1),0)),tf.where(tf.equal(tf.reduce_sum(active_mask_split,axis=1),0)),tf.shape(norm_factor)],message="trans",summarize=1000)

                    return XE, 0.0,0.0,0.0,0.0

                def steady_one(elems):
                    tensor,x,active_mask,key_mask,seq_len = elems

                    x = x[:seq_len,:]
                    tensor = tensor[:seq_len,:]
                    active_mask = active_mask[:seq_len,:]
                    key_mask = key_mask[:seq_len,:]

                    tensor,active_mask,key_mask,_ = self.split_steady(x,tensor,active_mask,key_mask)
                    XE = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor, labels=key_mask)*active_mask
                    XE = tf.reduce_sum(XE)/tf.reduce_sum(active_mask)

                    return XE, 0.0,0.0,0.0,0

                def steady_one_norm(elems):
                    tensor,x,seq_len,active_mask,key_mask = elems

                    x = x[:seq_len,:]
                    tensor = tensor[:seq_len,:]
                    active_mask = active_mask[:seq_len,:]
                    key_mask = key_mask[:seq_len,:]

                    tensor_split,active_mask_split,key_mask_split,_ = self.split_steady(x,tensor,active_mask,key_mask)
                    XE = tf.nn.sigmoid_cross_entropy_with_logits(logits=tensor_split, labels=key_mask_split)*active_mask_split
                    norm_factor = tf.expand_dims(tf.reduce_sum(key_mask_split,axis=[1]),axis=1)
                    XE = XE/norm_factor
                    # XE = tf.where(tf.is_nan(XE), tf.zeros_like(XE), XE)
                    XE= tf.reduce_sum(XE)/tf.reduce_sum(active_mask_split)
                    # XE = tf.Print(XE,[tf.where(tf.equal(tf.reduce_sum(key_mask_split,axis=1),0)),tf.where(tf.equal(tf.reduce_sum(active_mask_split,axis=1),0)),tf.shape(norm_factor)],message="steady",summarize=1000)

                    return XE, 0.0,0.0,0.0,0



                n_notes = self.n_notes
                n_steps = self.n_steps
                x = self.inputs
                y = self.labels
                seq_lens = self.seq_lens
                if hasattr(self, 'scheduled_sampling') and self.scheduled_sampling:
                    pred = self.prediction_sched_samp
                else:
                    pred = self.prediction
                key_masks = self.key_masks
                thresh_key = self.thresh_key


                key_masks = tf.cast(tf.greater(key_masks,thresh_key),tf.float32)

                label_mask = tf.cast(tf.abs(1-y),tf.float32)

                # label_mask = label_mask*tf.cast(tf.abs(1-x[:,:-1,:]),tf.float32)
                length_mask = tf.sequence_mask(seq_lens,maxlen=n_steps)
                length_mask = tf.expand_dims(length_mask,-1)
                length_mask = tf.cast(tf.tile(length_mask,[1,1,n_notes]),tf.float32)
                XE_mask = label_mask*length_mask
                pred_masked = pred*XE_mask


                key_XE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=key_masks)*XE_mask,axis=[1,2])/tf.reduce_sum(XE_mask,axis=[1,2])


                #Key cross_entropy on transitions
                output = tf.map_fn(transitions_one,(pred_masked,x,XE_mask,key_masks,seq_lens),dtype=(tf.float32,tf.int32,tf.float32,tf.float32,tf.int32))
                key_XE_tr = output[0]


                #Key cross_entropy on steady state
                output = tf.map_fn(steady_one,(pred_masked,x,XE_mask,key_masks,seq_lens),dtype=(tf.float32,tf.int32,tf.float32,tf.float32,tf.int32))
                key_XE_ss = output[0]


                #NORMALISED XE_k
                norm_factor = tf.expand_dims(tf.reduce_sum(key_masks,axis=[2]),axis=2)
                key_XE_n = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=key_masks)*XE_mask/norm_factor
                key_XE_n = tf.where(tf.is_nan(key_XE_n), tf.zeros_like(key_XE_n), key_XE_n)
                key_XE_n = tf.reduce_sum(key_XE_n,axis=[1,2])/tf.reduce_sum(XE_mask,axis=[1,2])

                #NORMALISED XE_k,tr
                output = tf.map_fn(transitions_one_norm,(pred_masked,x,seq_lens,XE_mask,key_masks),dtype=(tf.float32,tf.int32,tf.float32,tf.float32,tf.float32))
                key_XE_tr_n = output[0]


                #NORMALISED XE_k,ss
                output = tf.map_fn(steady_one_norm,(pred_masked,x,seq_lens,XE_mask,key_masks),dtype=(tf.float32,tf.int32,tf.float32,tf.float32,tf.float32))
                key_XE_ss_n = output[0]


                cross_entropy_key_masked = [key_XE,key_XE_tr,key_XE_ss,key_XE_n,key_XE_tr_n,key_XE_ss_n]

                self._cross_entropy_key2 = cross_entropy_key_masked
        return self._cross_entropy_key2

    @property
    def cross_entropy_key(self):
        if self._cross_entropy_key is None:
            with tf.device(self.device_name):


                key_XE,key_XE_tr,key_XE_ss,key_XE_n,key_XE_tr_n,key_XE_ss_n = self.cross_entropy_key2

                key_XE = tf.reduce_mean(key_XE)

                #Key cross_entropy on transitions
                key_XE_tr = tf.reduce_mean(tf.gather(key_XE_tr,tf.where(tf.logical_not(tf.is_nan(key_XE_tr)))))

                #Key cross_entropy on steady state
                key_XE_ss = tf.reduce_mean(tf.gather(key_XE_ss,tf.where(tf.logical_not(tf.is_nan(key_XE_ss)))))
                # key_XE_ss = tf.where(tf.is_nan(key_XE_ss), tf.zeros_like(key_XE_ss), key_XE_ss)


                #NORMALISED XE_k
                key_XE_n = tf.reduce_mean(key_XE_n)

                #NORMALISED XE_k,tr
                key_XE_tr_n = tf.reduce_mean(tf.gather(key_XE_tr_n,tf.where(tf.logical_not(tf.is_nan(key_XE_tr_n)))))

                #NORMALISED XE_k,ss
                key_XE_ss_n = tf.reduce_mean(tf.gather(key_XE_ss_n,tf.where(tf.logical_not(tf.is_nan(key_XE_ss_n)))))


                cross_entropy_key_masked = [key_XE,key_XE_tr,key_XE_ss,key_XE_n,key_XE_tr_n,key_XE_ss_n]

                self._cross_entropy_key = cross_entropy_key_masked
        return self._cross_entropy_key

    @property
    def combined_metric2(self):
        if self._combined_metric2 is None:
            with tf.device(self.device_name):
                XE_tr = self.cross_entropy_transition2
                XE_ss = self.cross_entropy_steady2
                XE_k = self.cross_entropy_key2
                XE_ktr = XE_k[1]
                XE_kss = XE_k[2]

                combined_metric = tf.sqrt((XE_tr+XE_ss)*(XE_ktr+XE_kss))
                # combined_metric = tf.Print(combined_metric,[XE_tr,XE_ss,XE_ktr,XE_kss,(XE_tr+XE_ss),(XE_ktr+XE_kss),combined_metric],message='combined')

                self._combined_metric2 = combined_metric
        return self._combined_metric2

    @property
    def combined_metric(self):
        if self._combined_metric is None:
            with tf.device(self.device_name):
                combined_metric = self.combined_metric2

                combined_metric = tf.gather(combined_metric,tf.where(tf.logical_not(tf.is_nan(combined_metric))))
                combined_metric = tf.reduce_mean(combined_metric)


                self._combined_metric = combined_metric
        return self._combined_metric

    @property
    def combined_metric_norm2(self):
        if self._combined_metric_norm2 is None:
            with tf.device(self.device_name):
                XE_tr = self.cross_entropy_transition2
                XE_ss = self.cross_entropy_steady2
                XE_k = self.cross_entropy_key2
                XE_ktr = XE_k[4]
                XE_kss = XE_k[5]

                combined_metric_norm = tf.sqrt((XE_tr+XE_ss)*(XE_ktr+XE_kss))


                self._combined_metric_norm2 = combined_metric_norm
        return self._combined_metric_norm2

    @property
    def combined_metric_norm(self):
        if self._combined_metric_norm is None:
            with tf.device(self.device_name):
                combined_metric = self.combined_metric_norm2

                combined_metric = tf.gather(combined_metric,tf.where(tf.logical_not(tf.is_nan(combined_metric))))
                combined_metric = tf.reduce_mean(combined_metric)


                self._combined_metric_norm = combined_metric
        return self._combined_metric_norm

    @property
    def gamma(self):
        """
        Focal loss gamma parameter
        """
        if self._gamma is None:
            with tf.device(self.device_name):
                suffix = self.suffix
                self._gamma = tf.placeholder_with_default(2.0,shape=[],name="gamma"+suffix)
        return self._gamma

    @property
    def focal_loss(self):
        """
        Focal loss: decreases the importance of correctly detected bins, focuses on mistakes
        """
        if self._focal_loss is None:
            with tf.device(self.device_name):

                y = self.labels
                pred_sigm = self.pred_sigm
                y_inv = 1-y
                p_t = tf.abs(y_inv - pred_sigm)
                if self.scheduled_sampling:
                    pred = self.prediction_sched_samp
                else:
                    pred = self.prediction
                logits = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y)
                focal_loss = tf.reduce_mean(tf.pow(p_t,self.gamma)*logits)

                self._focal_loss = focal_loss
        return self._focal_loss


    @property
    def loss(self):
        if self._loss is None:
            with tf.device(self.device_name):

                XE = self.cross_entropy

                if self.loss_type == 'focal_loss':
                    print('Use Focal Loss')
                    loss = self.focal_loss
                elif self.loss_type == 'XE':
                    print('Use Cross-Entropy Loss')
                    loss = XE #+ 0.1*cross_entropy_trans
                elif self.loss_type == 'combined':
                    print('Use Combined metric Loss')
                    loss = self.combined_metric


                elif self.loss_type == 'combined_norm':
                    print('Use Normalised Combined metric Loss')
                    loss = self.combined_metric_norm

                elif self.loss_type == "XEtr_XEss":
                    print('Use (XE_tr+XE_ss) Loss')
                    XE_tr = self.cross_entropy_transition2
                    XE_ss = self.cross_entropy_steady2

                    no_nan_mask = tf.logical_or(tf.is_nan(XE_tr),tf.is_nan(XE_tr))
                    no_nan_mask = tf.logical_not(no_nan_mask)

                    XE_tr = tf.gather(XE_tr,tf.where(no_nan_mask))
                    XE_ss = tf.gather(XE_ss,tf.where(no_nan_mask))

                    loss = tf.reduce_mean(XE_tr+XE_ss)
                else:
                    raise ValueError("loss_type value not understood: "+str(self.loss_type) )

                self._loss = loss
        return self._loss



    @property
    def optimize(self):
        if self._optimize is None:
            with tf.device(self.device_name):

                loss = self.loss

                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gvs = optimizer.compute_gradients(loss)
                if self.grad_clip is not None:
                    print("Gradient clipping",self.grad_clip)
                    capped_gvs = [(tf.clip_by_value(grad, -float(self.grad_clip), float(self.grad_clip)), var) for grad, var in gvs]
                    # grad_check = tf.check_numerics([x[0] for x in capped_gvs])
                    # with tf.control_dependencies([grad_check]):
                    train_op = optimizer.apply_gradients(capped_gvs)
                else:
                    print('No Gradient clipping')
                    train_op = optimizer.apply_gradients(gvs)
                train_op = optimizer.minimize(loss)
                self._optimize = train_op
        return self._optimize

    def _run_by_batch(self,sess,op,feed_dict,batch_size,mean=True):
        """
        Evaluate a node by batch, splitting the dataset.
        Currently, only works with cross_entropy2, so not a generic operator.
        """
        suffix = self.suffix
        x = self.inputs

        y = self.labels
        seq_len = self.seq_lens
        batch_size_ph = self.batch_size


        dataset = feed_dict[x]
        target = feed_dict[y]
        len_list = feed_dict[seq_len]
        if self.scheduled_sampling == 'mix':
            ac_out_ph = self.acoustic_outputs
            ac_out = feed_dict[ac_out_ph]
        if self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
            keys_ph = self.key_masks
            keys = feed_dict[keys_ph]


        no_of_batches = int(np.ceil(float(len(dataset))/batch_size))
        #crosses = np.zeros([dataset.shape[0]])
        #results = np.empty(dataset.shape)
        results = []
        ptr = 0
        for j in range(no_of_batches):
            batch_x = dataset[ptr:ptr+batch_size]
            batch_y = target[ptr:ptr+batch_size]
            batch_len_list = len_list[ptr:ptr+batch_size]
            feed_dict.update({x:batch_x,y:batch_y,seq_len:batch_len_list,batch_size_ph:batch_x.shape[0]})
            if self.scheduled_sampling == 'mix':
                batch_ac_out = ac_out[ptr:ptr+batch_size]
                feed_dict.update({ac_out_ph:batch_ac_out})
            if self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
                batch_keys = keys[ptr:ptr+batch_size]
                feed_dict.update({keys_ph:batch_keys})


            ptr += batch_size
            result_batch = sess.run(op, feed_dict=feed_dict)
            results = np.append(results,result_batch)
        if mean:
            return np.mean(results)
        else :
            return results


    def extract_data(self,dataset,subset):
        """
        Get NumPy tensors for input, target and sequence lengths from Dataset object
        in the right format to be given as values to placeholders.
        """
        if self.pitchwise:
            data = []
            targets = []
            lengths = []

            generator = dataset.get_pitchwise_dataset_generator('valid',50,int((self.n_notes-1)/2.0),self.chunks)
            for input, target, len in generator:
                data+= [np.squeeze(a,axis=0) for a in np.split(input,input.shape[0],axis=0)]
                targets += [np.squeeze(a,axis=0) for a in np.split(target,target.shape[0],axis=0)]
                lengths += [np.squeeze(a,axis=0) for a in np.split(len,len.shape[0],axis=0)]

            data = np.array(data)
            targets = np.array(targets)
            lengths = np.array(lengths)

            data = np.transpose(data,[0,2,1])

            targets = np.transpose(targets,[0,2,1])
            output = [data, targets, lengths]

        else:
            chunks = self.chunks

            if chunks:
                if self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
                    data, targets, lengths, key_masks  = dataset.get_dataset_chunks_no_pad(subset,chunks,with_keys=True)
                    idx = dataset.check_data(data,lengths)
                    data = [data[i] for i in idx]
                    targets = [targets[i] for i in idx]
                    lengths = [lengths[i] for i in idx]
                    key_masks = [key_masks[i] for i in idx]
                else:
                    data, targets, lengths  = dataset.get_dataset_chunks_no_pad(subset,chunks)
            else :
                data, targets, lengths = dataset.get_dataset(subset)


            data = self._transpose_data(data)
            targets = self._transpose_data(targets)
            output = [data, targets, lengths]
            if self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
                key_masks = self._transpose_data(key_masks)
                output += [key_masks]

        return output

    def initialize_training(self,save_path,train_param,sess=None):
        """
        Prepare everything for training.
        Creates summaries, session if not already given, initialises variables,
        creates savers.
        """
        optimizer = self.optimize
        cross_entropy = self.cross_entropy
        precision = self.precision
        recall = self.recall
        f_measure = self.f_measure

        ckpt_save_path = os.path.join("./ckpt/",save_path)
        summ_save_path = os.path.join("./summ/",save_path)
        safe_mkdir(ckpt_save_path)

        init = tf.global_variables_initializer()
        # check_op = tf.add_check_numerics_ops()
        if sess is None:
            init = tf.global_variables_initializer()
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(init)

        else:
            variables_to_initialize = sess.run(tf.report_uninitialized_variables())
            var_list = []

            for var in tf.global_variables():
                for var_to_init in variables_to_initialize:
                    if var_to_init.decode("utf-8")  in var.name:
                        var_list += [var]
            init = tf.variables_initializer(var_list)
            sess.run(init)

        if train_param['summarize']:
            safe_mkdir(summ_save_path,clean=True)
            tf.summary.scalar('cross entropy epoch',cross_entropy,collections=['epoch'])
            tf.summary.scalar('precision epoch',tf.reduce_mean(precision),collections=['epoch'])
            tf.summary.scalar('recall epoch',tf.reduce_mean(recall),collections=['epoch'])
            tf.summary.scalar('f_measure epoch',tf.reduce_mean(f_measure),collections=['epoch'])

            tf.summary.scalar('cross entropy batch',cross_entropy,collections=['batch'])
            tf.summary.scalar('precision batch',tf.reduce_mean(precision),collections=['batch'])
            tf.summary.scalar('recall batch',tf.reduce_mean(recall),collections=['batch'])
            tf.summary.scalar('f_measure batch',tf.reduce_mean(f_measure),collections=['batch'])

            summary_epoch = tf.summary.merge_all('epoch')
            summary_batch = tf.summary.merge_all('batch')
            train_writer = tf.summary.FileWriter(summ_save_path,
                                      sess.graph)


        if train_param['early_stop']:
            saver = [tf.train.Saver(max_to_keep=train_param['max_to_keep']), tf.train.Saver(max_to_keep=1)]
        else:
            saver = tf.train.Saver(max_to_keep=train_param['max_to_keep'])

        return sess, saver, train_writer, ckpt_save_path, summary_batch, summary_epoch#, check_op


    def perform_training(self,data,save_path,train_param,sess,saver,train_writer,
        ckpt_save_path,summary_batch, summary_epoch,n_batch=0,n_epoch=0):
        """
        Actually performs training steps.
        """

        optimizer = self.optimize
        cross_entropy = self.cross_entropy
        cross_entropy2= self.cross_entropy2
        precision = self.precision
        recall = self.recall
        f_measure = self.f_measure
        suffix = self.suffix
        batch_size_ph = self.batch_size
        sched_samp_p = self.sched_samp_p

        x = self.inputs
        y = self.labels
        seq_len = self.seq_lens
        if self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
            keys = self.key_masks



        print('Starting computations : '+str(datetime.now()))

        print("Total number of parameters:", getTotalNumParameters())
        if train_param['early_stop']:
            best_cross = np.inf
            epoch_since_best = 0
            saver_best = saver[1]
            saver = saver[0]

        epochs = train_param['epochs']
        print("Training for "+str(epochs)+" epochs")
        batch_size = train_param['batch_size']
        i = n_epoch

        if self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
            valid_data, valid_target, valid_lengths,valid_keys = self.extract_data(data,'valid')

        else:
            valid_data, valid_target, valid_lengths = self.extract_data(data,'valid')

        ## scheduled sampling strategy
        if self.scheduled_sampling:
            if train_param['schedule_shape'] == 'linear':
                schedule = np.linspace(1.0,0.0,train_param['schedule_duration'])
            elif train_param['schedule_shape'] == 'sigmoid':
                schedule = 1 / (1 + np.exp(np.linspace(-5.0,5.0,train_param['schedule_duration'])))
            else:
                raise ValueError('Schedule not understood: '+train_param['schedule_shape'])
            #Rescale schedule between 1 and end_val (instead of 1 and 0)
            end_value = train_param['schedule_end_value']
            schedule = (1-end_value)*schedule + end_value


        while i < n_epoch+epochs and epoch_since_best<train_param['early_stop_epochs']:
            start_epoch = datetime.now()
            ptr = 0
            # training_data, training_target, training_lengths = self.extract_data(data,'train')
            if self.pitchwise:
                train_data_generator = data.get_pitchwise_dataset_generator('train',batch_size,int((self.n_notes-1)/2.0),self.chunks)
            else:
                if self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
                    train_data_generator = data.get_dataset_generator('train',batch_size,self.chunks,with_names=True,with_keys=True,check_data=True)
                else:
                    train_data_generator = data.get_dataset_generator('train',batch_size,self.chunks)
            display_step = None



            for batch_data in train_data_generator:
                if self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
                    batch_x,batch_y, batch_lens, batch_names,batch_keys = batch_data
                    batch_keys = np.transpose(batch_keys,[0,2,1])
                    # print(batch_names)
                else:
                    batch_x,batch_y, batch_lens = batch_data
                batch_x = np.transpose(batch_x,[0,2,1])
                batch_y = np.transpose(batch_y,[0,2,1])



                ## Simple scheduled sampling
                if self.scheduled_sampling:
                    if i < train_param['schedule_duration']:
                        p = schedule[i]
                    else:
                        p= train_param['schedule_end_value']

                    if self.scheduled_sampling == 'mix':
                        batch_ac_out = batch_x[:,:-1,:]
                        batch_x = batch_y[:,:-1,:]
                        batch_y = batch_y[:,1:,:]

                        valid_ac_out = valid_data[:,:-1,:]
                        valid_x = valid_target[:,:-1,:]
                        valid_y = valid_target[:,1:,:]
                        ac_out = self.acoustic_outputs

                        feed_dict_optim = {x: batch_x, y: batch_y, ac_out: batch_ac_out,seq_len: batch_lens,batch_size_ph:batch_x.shape[0],sched_samp_p:p}
                        feed_dict_valid = {x: valid_x, y: valid_y, ac_out: valid_ac_out,seq_len: valid_lengths,batch_size_ph:batch_size,sched_samp_p:train_param['schedule_end_value']}
                    elif self.scheduled_sampling == 'self':
                        feed_dict_optim = {x: batch_x, y: batch_y, seq_len: batch_lens,batch_size_ph:batch_x.shape[0],sched_samp_p:p}
                        feed_dict_valid = {x: valid_data, y: valid_target, seq_len: valid_lengths,batch_size_ph:batch_size,sched_samp_p:train_param['schedule_end_value']}
                else:
                    feed_dict_optim = {x: batch_x, y: batch_y, seq_len: batch_lens, batch_size_ph:batch_x.shape[0]}
                    feed_dict_valid = {x: valid_data, y: valid_target, seq_len: valid_lengths,batch_size_ph:batch_size}
                if  self.loss_type in ['combined_norm','combined',"XEtr_XEss"]:
                    feed_dict_optim.update({keys: batch_keys})
                    feed_dict_valid.update({keys: valid_keys})


                loss, _ = sess.run([self.loss,optimizer], feed_dict=feed_dict_optim)
                # print(metrics)
                # if loss != loss:
                #     return

                if not display_step is None and j%display_step == 0 :
                    cross_batch = sess.run(cross_entropy, feed_dict=feed_dict_optim)
                    print("Batch "+str(j)+ ", Cross entropy = "+"{:.5f}".format(cross_batch))
                    if train_param['summarize']:
                        summary_b = sess.run(summary_batch,feed_dict=feed_dict_optim)
                        train_writer.add_summary(summary_b,global_step=n_batch)
                n_batch += 1


            cross = self._run_by_batch(sess,self.loss,feed_dict_valid,batch_size)
            if train_param['summarize']:
                summary_e = sess.run(summary_epoch,feed_dict=feed_dict_valid)
                train_writer.add_summary(summary_e, global_step=i)
            print("_________________")
            print("Epoch: " + str(i) + ", Cross Entropy = " + \
                          "{:.5f}".format(cross))
            end_epoch = datetime.now()
            print("Computation time =", str(end_epoch-start_epoch))

            #Check if cross is NaN, if so, stop computations
            if cross != cross :
                break

            # Save the variables to disk.
            # When using scheduled sampling: Only start early stopping once schedule is finished.
            if train_param['early_stop'] and (train_param['schedule_duration'] is None or (train_param['schedule_duration'] is not None and i>train_param['schedule_duration'])):
                if cross<best_cross:
                        saved = saver_best.save(sess, os.path.join(ckpt_save_path,"best_model.ckpt"),global_step=i)
                        best_cross = cross
                        epoch_since_best = 0
                else:
                    saved = saver.save(sess, os.path.join(ckpt_save_path,"model.ckpt"),global_step=i)
                    epoch_since_best += 1

            else:
                if i%train_param['save_step'] == 0 or i == epochs-1:
                    saved = saver.save(sess, os.path.join(ckpt_save_path,"model.ckpt"),global_step=i)
                else :
                    saved = saver.save(sess, os.path.join(ckpt_save_path,"model.ckpt"))
            print(("Model saved in file: %s" % saved))

            i += 1
            # Shuffle the dataset before next epoch
            data.shuffle_one('train')
            if not display_step is None:
                print("_________________")

        return n_batch, n_epoch+epochs


    def train(self, data, save_path, train_param,sess=None,n_batch=0,n_epoch=0):
        """
        Train a network.
        This function was split into 2 functions to allow modification of training
        parameters or dataset.
        For example: call initialize_training once, then call several times
        perform_training, with different datasets each time.
        """


        sess, saver, train_writer, ckpt_save_path, summary_batch, summary_epoch = self.initialize_training(save_path,train_param,sess=sess)

        n_batch,n_epoch  = self.perform_training(data,save_path,train_param,sess,saver,train_writer,ckpt_save_path,summary_batch, summary_epoch,n_batch=n_batch,n_epoch=n_epoch)

        print(("Optimization finished ! "+str(datetime.now())))

        return n_batch, n_epoch

    def load(self,save_path,model_path=None):
        """
        Load the parameters from a checkpoint file.
        'save_path' is a folder in which to look for a best_model (or the latest
        saved)
        'model_path' allows to specify which model exactly should be loaded
        (only used if not None)
        """

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        if model_path==None:
            folder = os.path.join("./ckpt/",save_path)
            path = None
            for file in os.listdir(folder):
                if 'best_model' in file and '.meta' in file:
                    path = os.path.join(folder,file.replace('.meta',''))
            if path is None:
                print("BEST MODEL NOT FOUND!")
                path = tf.train.latest_checkpoint(os.path.join("./ckpt/",save_path))
        else:
            path = model_path

        # Get variables in graph
        all_variables = tf.global_variables()
        var_list = dict(zip([var.name[:-2] for var in all_variables],all_variables)) #var.name[:-2] to remove ':0' at the end of the name

        # print('before')
        # for key, value in var_list.items():
        #     print(key,value)

        # Get variables in checkpoint
        saved_vars = tf.train.list_variables(path)
        saved_vars_names = [v[0] for v in saved_vars]

        # print('in_checkpoint')
        # for item in saved_vars_names:
        #     print(item)

        # Match names (if trained without scheduled sampling, tested with, or vice versa)
        if self.scheduled_sampling:
            if 'W' in saved_vars_names and 'b' in saved_vars_names:
                var_list.update({'b':var_list['rnn/output_layer/bias'],
                                 'W':var_list['rnn/output_layer/kernel']})
                var_list.pop('rnn/output_layer/bias')
                var_list.pop('rnn/output_layer/kernel')
        else:
            if 'rnn/output_layer/bias' in saved_vars_names and 'rnn/output_layer/kernel' in saved_vars_names:
                var_list.update({'rnn/output_layer/bias':var_list['b'],
                                 'rnn/output_layer/kernel':var_list['W']})
                var_list.pop('W')
                var_list.pop('b')
        if 'rnn/lstm_cell/weights' in saved_vars_names:
            var_list.update({'rnn/lstm_cell/weights':var_list['rnn/lstm_cell/kernel'],
                            'rnn/lstm_cell/biases':var_list['rnn/lstm_cell/bias']})
            var_list.pop('rnn/lstm_cell/kernel')
            var_list.pop('rnn/lstm_cell/bias')

        # print('after')
        # for key, value in var_list.items():
        #     print(key,value)

        saver = tf.train.Saver(var_list=var_list)

        print("Loading "+path)
        saver.restore(sess, path)


        return sess, saver

    def resume_training(self,load_path,data,save_path,train_param,model_path=None,n_batch=0,n_epoch=0):
        """
        Resume training by loading a model from data and continuing training
        """

        sess, saver = self.load(load_path,model_path)

        n_batch,n_epoch = self.train(data,save_path,train_param,sess=sess,n_batch=n_batch,n_epoch=n_epoch)
        return n_batch,n_epoch


    def run_prediction(self,dataset,len_list, save_path,model_path=None,sigmoid=False,sess=None):
        """
        Get predictions as Numpy matrix.
        If sess or saver are not provided (None), a model will be loaded from
        checkpoint. Otherwise, the provided session and saver will be used.
        """

        if sess==None:
            sess, _ = self.load(save_path,model_path)

        suffix = self.suffix
        x = self.inputs
        seq_len = self.seq_lens
        batch_size_ph = self.batch_size

        dataset = self._transpose_data(dataset)

        if self.scheduled_sampling:
            pred = self.prediction_sched_samp
            feed_dict ={x: dataset, seq_len: len_list,batch_size_ph:dataset.shape[0],self.sched_samp_p:1.0}
        else:
            pred = self.prediction
            feed_dict = {x: dataset, seq_len: len_list,batch_size_ph:dataset.shape[0]}

        if sigmoid:
            if self.with_onsets:
                pred = tf.nn.softmax(pred)
            else:
                pred=tf.sigmoid(pred)

        notes_pred = sess.run(pred, feed_dict = feed_dict)
        if self.with_onsets:
            output = np.transpose(notes_pred,[0,2,1,3])
        else:
            output = np.transpose(notes_pred,[0,2,1])

        return output

    def get_initial_state(self,sess,batch_size):
        """
        Returns a zero-filled initial state

        Parameters
        ----------
        sess
            A Tensorflow session, obtained with 'Model.load'
        batch_size
            number of initial states to get (mostl likely equal to branching factor)
        """
        init_states = sess.run(self.initial_state,{self.batch_size:batch_size})
        init_c = init_states.c
        init_h = init_states.h

        return list(zip(np.split(init_c,init_c.shape[0]),np.split(init_h,init_h.shape[0])))

    def run_one_step(self,hidden_states_in,samples,sess):
        """
        Perform one step of prediction: get new hidden state and output distribution

        Parameters
        ----------
        hidden_states_in
            list of couples (cell_state,hidden_neurons), of length batch_size.
        samples
            a Numpy array holding the samples to evaluate for current timestep.
            Should be of dimension: [batch_size,1,n_notes]
        sess
            A Tensorflow session, obtained with 'Model.load'

        Returns
        -------
        hidden_states_out
            Updated hidden states
        predictions
            likelihoods for next timestep
        """


        suffix = self.suffix
        pred = self.pred_sigm
        x = self.inputs
        seq_len = self.seq_lens
        batch_size_ph = self.batch_size
        initial_state = self.initial_state
        output_state = self.output_state

        c,h= list(zip(*hidden_states_in))
        c = np.squeeze(np.array(c), axis=1)
        h = np.squeeze(np.array(h), axis=1)

        hidden_states_in =  tf.nn.rnn_cell.LSTMStateTuple(c=c,h=h)

        len_list = np.full([len(samples)],samples.shape[1])

        if self.scheduled_sampling:
            feed_dict ={x: samples, seq_len: len_list,batch_size_ph:len(samples),initial_state:hidden_states_in,self.sched_samp_p:1.0}
        else:
            feed_dict ={x: samples, seq_len: len_list,batch_size_ph:len(samples),initial_state:hidden_states_in}

        predictions,hidden_states_out = sess.run([pred,output_state], feed_dict = feed_dict )

        c_out, h_out = hidden_states_out

        hidden_states_out = list(zip(np.split(c_out,c_out.shape[0]),np.split(h_out,h_out.shape[0])))

        return hidden_states_out,predictions

    def run_cross_entropy(self,dataset,len_list, save_path,n_model=None,batch_size=50,mean=True):
        """
        Get cross-entropy as Numpy matrix.
        If sess or saver are not provided (None), a model will be loaded from
        checkpoint. Otherwise, the provided session and saver will be used.
        If mean==True, returns an average over the dataset, otherwise,
        one value for each sequence.
        """

        sess, saver = self.load(save_path,n_model)
        cross_entropy = self.cross_entropy2

        suffix = self.suffix
        x = self.inputs
        seq_len = self.seq_lens
        y = self.labels

        target = dataset[:,:,1:]
        dataset = self._transpose_data(dataset[:,:,:-1])
        target = self._transpose_data(target)
#        print type(target)

        cross = self._run_by_batch(sess,cross_entropy,{x: dataset,y: target,seq_len: len_list,batch_size_ph:batch_size},batch_size,mean=mean)
        return cross



    def compute_eval_metrics_pred(self,data,targets,len_list,threshold,save_path,keys=None,n_model=None,sess=None,no_sched=False):
        """
        Compute averaged metrics over the dataset.
        If sess or saver are not provided (None), a model will be loaded from
        checkpoint. Otherwise, the provided session and saver will be used.

        """

        if sess==None:
            sess, _ = self.load(save_path,n_model)

        pred = self.pred_sigm
        cross = self.cross_entropy_length
        cross_tr = self.cross_entropy_transition
        F0 = tf.reduce_mean(self.f_measure)


        data = self._transpose_data(data)
        targets = self._transpose_data(targets)




        suffix = self.suffix
        x = self.inputs
        seq_len = self.seq_lens
        y = self.labels
        thresh = self.thresh
        batch_size_ph = self.batch_size

        sched_samp_p = self.sched_samp_p

        if self.scheduled_sampling == 'mix':
            acoustic_outputs = self.acoustic_outputs

        if keys is not None:
            keys_ph = self.key_masks
            Score = self.combined_metric
            keys = self._transpose_data(keys)

        #Metrics with perfect input

        crosses = []
        crosses_tr = []
        F_measures = []
        Scores = []

        if no_sched:
            sched_values = [1]
        else:
            sched_values = np.arange(1.0,0.0,-0.1)

        for i in sched_values:
            if self.scheduled_sampling == 'mix':
                feed_dict = {x: targets[:,:-1,:], seq_len: len_list, y: targets[:,1:,:], acoustic_outputs:data[:,:-1,:],thresh: threshold,batch_size_ph:data.shape[0],sched_samp_p:i}
            else:
                feed_dict = {x: data, seq_len: len_list, y: targets, thresh: threshold,batch_size_ph:data.shape[0],sched_samp_p:i}

            if keys is not None:
                feed_dict.update({keys_ph:keys})
                cross_GT, cross_tr_GT, F_measure_GT, Score_GT = sess.run([cross, cross_tr, F0,Score], feed_dict = feed_dict)
            else:
                cross_GT, cross_tr_GT, F_measure_GT = sess.run([cross, cross_tr, F0], feed_dict = feed_dict)

            print(sess.run([cross, cross_tr, self.cross_entropy_steady, self.cross_entropy_length, self.cross_entropy_key, Score, F0], feed_dict = feed_dict ))

            crosses += [cross_GT]
            crosses_tr += [cross_tr_GT]
            F_measures += [F_measure_GT]


            if keys is not None:
                Scores += [Score_GT]


        output = [crosses,crosses_tr,F_measures]

        if keys is not None:
            output += [Scores]


        # #Metrics with thresholded input
        # sampled_frames = (preds>0.5).astype(int)
        # data[:,1:,:]=sampled_frames[:,:-1,:]
        # cross_th, cross_tr_th, F_measure_th = sess.run([cross, cross_tr, F0], feed_dict = {x: data, seq_len: len_list, y: targets, thresh: threshold,batch_size_ph:dataset.shape[0]} )



        return output

    def build_graph(self):
        self.cross_entropy
        return



def sample(probs,outshape=None):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    if outshape is None:
        output = np.floor(probs + np.random.uniform(0, 1,probs.shape))
    else:
        output = np.floor(probs + np.random.uniform(0, 1,outshape))
    return output



def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)



def getTotalNumParameters():
    '''
    Returns the total number of parameters contained in all trainable variables
    :return: Number of parameters (int)
    '''
    total_parameters = 0
    for variable in tf.trainable_variables():
        # print('------------')
        # print(variable.name)
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def make_model_from_dataset(dataset,model_param):
    """
    Initialise model_param dictionary given a Dataset object
    """

    n_notes = dataset.get_n_notes()
    if model_param['chunks']:
        n_steps = model_param['chunks']-1
    else:
        n_steps = dataset.get_len_files()-1

    if not model_param['pitchwise']:
        model_param['n_notes']=n_notes
    model_param['n_steps']=n_steps

    return Model(model_param)

def make_save_path(base_path,model_param,rep=None):
    """
    Systematically creates a save_path given model parameters
    """

    n_hidden = model_param['n_hidden']
    learning_rate = model_param['learning_rate']

    if type(n_hidden) == int:
        hidden_learning_path= str(n_hidden)+"_"+str(learning_rate)
    else:
        n_hidden_string = '-'.join([str(n) for n in n_hidden])
        hidden_learning_path= n_hidden_string+"_"+str(learning_rate)

    if model_param['chunks']:
        chunk_path = "_ch"+str(model_param['chunks'])
    else:
        chunk_path = ""

    if rep!=None:
        rep_path = "_"+str(rep)
    else:
        rep_path=""

    return os.path.join(base_path,hidden_learning_path+chunk_path+rep_path+"/")


def make_model_param():
    """
    Create a default 'model_param'
    """
    model_param = {}

    model_param['n_hidden']=128
    model_param['learning_rate']=0.01
    model_param['n_notes']=88
    model_param['n_steps']=300
    model_param['loss_type']= 'XE'
    model_param['cell_type']= 'LSTM'
    model_param['grad_clip']=None
    model_param['pitchwise']=False
    model_param['scheduled_sampling'] = False
    model_param['sampl_mix_weight']= None
    model_param['with_onsets']=False

    model_param['chunks']=None
    model_param['device_name']="/gpu:0"
    model_param['suffix']=""

    return model_param

def make_train_param():
    """
    Create a default 'train_param'
    """
    train_param = {}

    train_param['epochs']=20
    train_param['batch_size']=50

    train_param['display_per_epoch']=10,
    train_param['save_step']=1
    train_param['max_to_keep']=5
    train_param['summarize']=True
    train_param['early_stop']=True
    train_param['early_stop_epochs']=15
    train_param['schedule_shape']=None
    train_param['schedule_duration']=None
    train_param['schedule_end_value']=None


    return train_param





# data = Dataset()
# data.load_data('data/test_dataset/',
#         fs=4,max_len=None,note_range=[21,109],quant=True,length_of_chunks=30)
# data.transpose_all()
#
# model_param = make_model_param()
# train_param = make_train_param()
# train_param['batch_size']=5
# train_param['display_per_epoch']=5
#
# model = make_model_from_dataset(data,model_param)
# model.train(data,save_path="./tmp/crop_unnorm/",train_param=train_param)

# model.run_prediction(dataset="loul",save_path="./tmp/crop_unnorm/",n_model=19)
# print model.run_cross_entropy(dataset="loul",save_path="./tmp/crop_unnorm/",n_model=99)





class ModLSTMCell(tf.contrib.rnn.RNNCell):
    """Modified LSTM Cell """

    def __init__(self, num_units, initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=2, dtype=tf.float32), wform = 'diagonal'):
        self._num_units = num_units
        self.init = initializer
        self.wform = wform

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"

            c, h = state
            init = self.init
            self.L1 = inputs.get_shape().as_list()[1]

            mats, biases = self.get_params_parallel()
            if self.wform == 'full' or self.wform == 'diag_to_full':

                res = tf.matmul(tf.concat([h,inputs],axis=1),mats)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1)
            elif self.wform == 'diagonal':
                h_concat = tf.concat([h,h,h,h],axis=1)

                W_res = tf.multiply(h_concat,mats[0])

                U_res = tf.matmul(inputs,mats[1])

                res = tf.add(W_res,U_res)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1)

            elif self.wform == 'constant':

                h_concat = tf.concat([h,h,h,h],axis=1)

                U_res = tf.matmul(inputs,mats)

                res = tf.add(h_concat,U_res)
                res_wbiases = tf.nn.bias_add(res, biases)

                i,j,f,o = tf.split(res_wbiases,num_or_size_splits=4,axis=1)


            new_c = (c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i)*tf.nn.tanh(j))

            new_h = tf.nn.tanh(new_c) * tf.nn.sigmoid(o)
            new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state


    def get_params_parallel(self):
        if self.wform == 'full':
            mats = tf.get_variable("mats",
                    shape = [self._num_units+self.L1,self._num_units*4],
                    initializer = self.init )
            biases = tf.get_variable("biases",
                    shape = [self._num_units*4],
                    initializer = self.init )
        elif self.wform == 'diagonal':
            Ws = tf.get_variable("Ws",
                    shape = [1,self._num_units*4],
                    initializer = self.init )
            Umats = tf.get_variable("Umats",
                    shape = [self.L1,self._num_units*4],
                    initializer = self.init )
            biases = tf.get_variable("biases",
                    shape = [self._num_units*4],
                    initializer = self.init )
            mats = [Ws, Umats]
        elif self.wform == 'constant':
            mats = tf.get_variable("mats",
                    shape = [self.L1,self._num_units*4],
                    initializer = self.init )
            biases = tf.get_variable("biases",
                    shape = [self._num_units*4],
                    initializer = self.init )
        elif self.wform == 'diag_to_full':
            #get the current variable scope
            var_scope = tf.get_variable_scope().name.replace('rnn2/','')

            #first filtering
            vars_to_use = [var for var in self.init if var_scope in var[0]]

            #next, assign the variables
            for var in vars_to_use:
                if '/Ws' in var[0]:
                    Ws = np.split(var[1], indices_or_sections = 4, axis = 1)
                    diag_Ws = np.concatenate([np.diag(w.squeeze()) for w in Ws], axis = 1)

                elif '/Umats' in var[0]:
                    Us = var[1]

                elif '/biases' in var[0]:
                    biases_np = var[1]

            mats_np = np.concatenate([diag_Ws, Us], axis = 0)
            mats_init = tf.constant_initializer(mats_np)

            mats = tf.get_variable("mats",
                    shape = [self._num_units+self.L1,self._num_units*4],
                    initializer = mats_init )

            biases_init = tf.constant_initializer(biases_np)
            biases = tf.get_variable("biases",
                    shape = [self._num_units*4],
                    initializer = biases_init )




        return mats, biases
