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
            setattr(self,key,value)

        self._inputs = None
        self._seq_lens = None
        self._labels = None
        self._thresh = None


        self._prediction = None
        self._pred_sigm = None
        self._last_pred_sigm = None
        self._pred_thresh = None
        self._cross_entropy = None
        self._cross_entropy2 = None
        self._optimize = None
        self._tp = None
        self._fp = None
        self._fn = None
        self._precision = None
        self._recall = None
        self._f_measure = None


        #Call to create the graph
        self.cross_entropy


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
    def prediction(self):
        """
        Logit predictions: x[t] given x[0:t]
        """
        if self._prediction is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_classes = n_notes
                n_steps = self.n_steps
                n_hidden = self.n_hidden
                suffix = self.suffix


                x = self.inputs
                seq_len = self.seq_lens
                dropout = tf.placeholder_with_default(1.0, shape=(), name="dropout"+suffix)

                W = tf.Variable(tf.truncated_normal([n_hidden,n_classes]),name="W"+suffix)
                b = tf.Variable(tf.truncated_normal([n_classes]),name="b"+suffix)

                cell = tf.contrib.rnn.LSTMCell(n_hidden,state_is_tuple=True,forget_bias = 1.0)

                #We don't take into account sequence length because doing so
                #causes Tensorflow to output weird results
                outputs, _ = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32,time_major=False)#,sequence_length=seq_len)


                outputs = tf.reshape(outputs,[-1,n_hidden])
                pred = tf.matmul(outputs,W) + b

                pred = tf.reshape(pred,[-1,n_steps,n_notes])
                #drop last prediction of each sequence (you don't have ground truth for this one)
                pred = pred[:,:n_steps-1,:]
                self._prediction = pred
        return self._prediction


    @property
    def pred_sigm(self):
        """
        Sigmoid predictions: x[t] given x[0:t]
        """
        if self._pred_sigm is None:
            with tf.device(self.device_name):
                pred = self.prediction
                pred = tf.sigmoid(pred)
                self._pred_sigm = pred
        return self._pred_sigm

    @property
    def last_pred_sigm(self):
        """
        Last sigmoid prediction: N=len(x); x[N-1] given x[0:N-1]
        """
        if self._last_pred_sigm is None:
            with tf.device(self.device_name):
                pred_sigm = self.pred_sigm
                self._last_pred_sigm = pred_sigm[:,-1,:]
        return self._last_pred_sigm

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
            n_notes = self.n_notes
            n_steps = self.n_steps
            suffix = self.suffix

            y = tf.placeholder("float", [None,n_steps-1,n_notes],name="y"+suffix)

            self._labels = y
        return self._labels

    @property
    def cross_entropy(self):
        """
        Mean cross entropy
        """
        if self._cross_entropy is None:
            with tf.device(self.device_name):
                y = self.labels

                cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=y))
                self._cross_entropy = cross_entropy
        return self._cross_entropy


    @property
    def cross_entropy2(self):
        """
        Cross entropy as a vector of length batch_size
        """
        if self._cross_entropy2 is None:
            with tf.device(self.device_name):
                n_notes = self.n_notes
                n_steps = self.n_steps
                suffix = self.suffix
                y = self.labels
                cross_entropy2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=y),axis=[1,2])
                self._cross_entropy2 = cross_entropy2
        return self._cross_entropy2


    @property
    def optimize(self):
        """
        Optimiser. Evaluate that node to train the network.
        """
        if self._optimize is None:
            with tf.device(self.device_name):
                cross_entropy = self.cross_entropy
                loss = cross_entropy
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)
                self._optimize = optimizer
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

        if y in feed_dict:
            dataset = feed_dict[x]
            target = feed_dict[y]
            len_list = feed_dict[seq_len]
        else:
            dataset = feed_dict[x]

        no_of_batches = int(np.ceil(float(len(dataset))/batch_size))
        #crosses = np.zeros([dataset.shape[0]])
        #results = np.empty(dataset.shape)
        results = []
        ptr = 0
        for j in range(no_of_batches):
            if y in feed_dict:
                batch_x = dataset[ptr:ptr+batch_size]
                batch_y = target[ptr:ptr+batch_size]
                batch_len_list = len_list[ptr:ptr+batch_size]
                feed_dict={x: batch_x, y: batch_y,seq_len: batch_len_list}
            else :
                batch_x = dataset[ptr:ptr+batch_size]
                feed_dict={x: batch_x}
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

        chunks = self.chunks

        if chunks:
            data_raw, lengths = dataset.get_dataset_chunks_no_pad(subset,chunks)
        else :
            data_raw, lengths = dataset.get_dataset(subset)

        data = self._transpose_data(data_raw)
        target = self._transpose_data(ground_truth(data_raw))

        return data, target, lengths

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
        if sess is None:
            init = tf.global_variables_initializer()
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(init)

        else:
            variables_to_initialize = sess.run(tf.report_uninitialized_variables())
            var_list = []
            for var in tf.global_variables():
                for var_to_init in variables_to_initialize:
                    if var_to_init in var.name:
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

        return sess, saver, train_writer, ckpt_save_path, summary_batch, summary_epoch


    def perform_training(self,data,save_path,train_param,sess,saver,train_writer,ckpt_save_path,summary_batch, summary_epoch,n_batch=0,n_epoch=0):
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

        x = self.inputs
        y = self.labels
        seq_len = self.seq_lens

        drop = tf.get_default_graph().get_tensor_by_name("dropout"+suffix+":0")

        print('Starting computations : '+str(datetime.now()))

        print("Total number of parameters:", getTotalNumParameters())
        if train_param['early_stop']:
            best_cross = np.inf
            epoch_since_best = 0
            saver_best = saver[1]
            saver = saver[0]

        epochs = train_param['epochs']
        batch_size = train_param['batch_size']
        i = n_epoch
        while i < n_epoch+epochs and epoch_since_best<train_param['early_stop_epochs']:
            start_epoch = datetime.now()
            ptr = 0

            training_data, training_target, training_lengths = self.extract_data(data,'train')
            valid_data, valid_target, valid_lengths = self.extract_data(data,'valid')


            n_files = training_data.shape[0]
            no_of_batches = int(np.ceil(float(n_files)/batch_size))

            display_step = max(int(round(float(no_of_batches)/train_param['display_per_epoch'])),1)



            for j in range(no_of_batches):
                batch_x = training_data[ptr:ptr+batch_size]
                batch_y = training_target[ptr:ptr+batch_size]
                batch_lens = training_lengths[ptr:ptr+batch_size]

                ptr += batch_size

                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, seq_len: batch_lens, drop: train_param['dropout']})

                if j%display_step == 0 :
                    cross_batch = sess.run(cross_entropy, feed_dict={x: batch_x, y: batch_y, seq_len: batch_lens})
                    print("Batch "+str(j)+ ", Cross entropy = "+"{:.5f}".format(cross_batch))
                    if train_param['summarize']:
                        summary_b = sess.run(summary_batch,feed_dict={x: batch_x, y: batch_y, seq_len: batch_lens})
                        train_writer.add_summary(summary_b,global_step=n_batch)
                n_batch += 1

            cross = self._run_by_batch(sess,cross_entropy2,{x: valid_data, y: valid_target, seq_len: valid_lengths},batch_size)
            if train_param['summarize']:
                summary_e = sess.run(summary_epoch,feed_dict={x: valid_data, y: valid_target, seq_len: valid_lengths})
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
            if train_param['early_stop']:
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

    def load(self,save_path,model_path):
        """
        Load the parameters from a checkpoint file.
        'save_path' is a folder in which to look for a best_model (or the latest
        saved)
        'model_path' allows to specify which model exactly should be loaded
        (only used if not None)
        """

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        saver = tf.train.Saver()
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


    def run_prediction(self,dataset,len_list, save_path,model_path=None,sigmoid=False,sess=None,saver=None):
        """
        Get predictions as Numpy matrix.
        If sess or saver are not provided (None), a model will be loaded from
        checkpoint. Otherwise, the provided session and saver will be used.
        """

        if sess==None or saver==None:
            sess, saver = self.load(save_path,model_path)

        suffix = self.suffix
        pred = self.prediction
        x = self.inputs
        seq_len = self.seq_lens

        dataset = self._transpose_data(dataset)

        notes_pred = sess.run(pred, feed_dict = {x: dataset, seq_len: len_list} )
        notes_pred = tf.transpose(notes_pred,[0,2,1])

        if sigmoid:
            notes_pred=tf.sigmoid(notes_pred)

        output = notes_pred.eval(session = sess)
        return output

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

        target = ground_truth(dataset)
        dataset = self._transpose_data(dataset)
        target = self._transpose_data(target)
#        print type(target)

        cross = self._run_by_batch(sess,cross_entropy,{x: dataset,y: target,seq_len: len_list},batch_size,mean=mean)
        return cross



    def compute_eval_metrics_pred(self,dataset,len_list,key_masks,threshold,save_path,batch_size=1,n_model=None,sess=None,saver=None):
        """
        Compute averaged metrics over the dataset.
        If sess or saver are not provided (None), a model will be loaded from
        checkpoint. Otherwise, the provided session and saver will be used.

        """

        # preds = self.run_prediction(dataset,len_list, save_path,n_model,sigmoid=True)
        # idx = preds[:,:,:] > threshold
        # preds_thresh = idx.astype(int)


        if sess==None and saver==None:
            sess, saver = self.load(save_path,n_model)

        cross = self.cross_entropy

        data = self._transpose_data(dataset)
        targets = self._transpose_data(ground_truth(dataset))
        k_masks = self._transpose_data(key_masks)

        prec = tf.reduce_mean(self.precision)
        rec = tf.reduce_mean(self.recall)
        F0 = tf.reduce_mean(self.f_measure)

        suffix = self.suffix
        x = self.inputs
        seq_len = self.seq_lens
        y = self.labels
        thresh = self.thresh


        cross, precision, recall, F_measure = sess.run([cross, prec, rec, F0], feed_dict = {x: data, seq_len: len_list, y: targets, thresh: threshold} )

        return F_measure, precision, recall, cross



def sample(probs):
    #Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return np.floor(probs + np.random.uniform(0, 1,probs.shape))

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)



def getTotalNumParameters():
    '''
    Returns the total number of parameters contained in all trainable variables
    :return: Number of parameters (int)
    '''
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            #print(dim)
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def make_model_from_dataset(dataset,model_param):
    """
    Initialise model_param dictionary given a Dataset object
    """

    n_notes = dataset.get_n_notes()
    if model_param['chunks']:
        n_steps = model_param['chunks']
    else:
        n_steps = dataset.get_len_files()

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
    model_param['batch_size']=50

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
    train_param['dropout']=1.0

    train_param['display_per_epoch']=10,
    train_param['save_step']=1
    train_param['max_to_keep']=5
    train_param['summarize']=True
    train_param['early_stop']=True
    train_param['early_stop_epochs']=15
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
