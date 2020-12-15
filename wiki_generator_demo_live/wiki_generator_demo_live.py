"""
We will use an LSTM RNN in TensorFlow that learns from wiki text to generate new text.

Dataset: https://metamind.io/research/the-wikitext-long-term-dependency-language-modeling-dataset/

100 million tokens extracted from articles on Wikipedia.

It will output the test input string plus 500 generated characters.
"""
import datetime
import numpy
import random 
import tensorflow

class WikiGenerator:
    """
    DOCSTRING
    """
    def __call__(self):
        text = open('wiki.test.raw').read()
        print('text length in number of characters:', len(text))
        print('head of text:')
        print(text[:1000])
        chars = sorted(list(set(text)))
        char_size = len(chars)
        print('number of characters:', char_size)
        print(chars)
        print('hello')
        char2id = dict((c, i) for i, c in enumerate(chars))
        id2char = dict((i, c) for i, c in enumerate(chars))
        len_per_section = 50
        skip = 2
        sections = list()
        next_chars = list()
        for i in range(len(text) - len_per_section, skip):
            sections.append(text[i: i + len_per_section])
            next_chars.append(text[i + len_per_section])
        X = numpy.zeros((len(sections), len_per_section, char_size))
        y = numpy.zeros((len(sections), char_size))
        for i, section in enumerate(sections):
            for j, char in enumerate(section):
                X[i, j, char2id[char]] = 1
            y[i, char2id[next_chars[i]]] = 1
        print(y)
        batch_size = 512
        max_steps = 72001
        log_every = 100
        save_every = 6000
        hidden_nodes = 1024
        test_start = 'I am thinking that'
        checkpoint_directory = 'ckpt'
        if tensorflow.gfile.Exists(checkpoint_directory):
            tensorflow.gfile.DeleteRecursively(checkpoint_directory)
        tensorflow.gfile.MakeDirs(checkpoint_directory)
        print('training data size:', len(X))
        print('approximate steps per epoch:', int(len(X)/batch_size))
        graph = tensorflow.Graph()
        with graph.as_default():
            global_step = tensorflow.Variable(0)
            data = tensorflow.placeholder(tensorflow.float32, [batch_size, len_per_section, char_size])
            labels = tensorflow.placeholder(tensorflow.float32, [batch_size, char_size])
            #Input gate: weights for input, weights for previous output, and bias
            w_ii = tensorflow.Variable(tensorflow.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
            w_io = tensorflow.Variable(tensorflow.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
            b_i = tensorflow.Variable(tensorflow.zeros([1, hidden_nodes]))
            #Forget gate: weights for input, weights for previous output, and bias
            w_fi = tensorflow.Variable(tensorflow.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
            w_fo = tensorflow.Variable(tensorflow.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
            b_f = tensorflow.Variable(tensorflow.zeros([1, hidden_nodes]))
            #Output gate: weights for input, weights for previous output, and bias
            w_oi = tensorflow.Variable(tensorflow.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
            w_oo = tensorflow.Variable(tensorflow.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
            b_o = tensorflow.Variable(tensorflow.zeros([1, hidden_nodes]))
            #Memory cell: weights for input, weights for previous output, and bias
            w_ci = tensorflow.Variable(tensorflow.truncated_normal([char_size, hidden_nodes], -0.1, 0.1))
            w_co = tensorflow.Variable(tensorflow.truncated_normal([hidden_nodes, hidden_nodes], -0.1, 0.1))
            b_c = tensorflow.Variable(tensorflow.zeros([1, hidden_nodes]))
            # LTSM
            output = tensorflow.zeros([batch_size, hidden_nodes])
            state = tensorflow.zeros([batch_size, hidden_nodes])
            for i in range(len_per_section):
                output, state = self.lstm(data[:, i, :], output, state)
                if i == 0:
                    outputs_all_i = output
                    labels_all_i = data[:, i+1, :]
                elif i != len_per_section - 1:
                    outputs_all_i = tensorflow.concat(0, [outputs_all_i, output])
                    labels_all_i = tensorflow.concat(0, [labels_all_i, data[:, i+1, :]])
                else:
                    outputs_all_i = tensorflow.concat(0, [outputs_all_i, output])
                    labels_all_i = tensorflow.concat(0, [labels_all_i, labels])
            # classifier
            w = tensorflow.Variable(tensorflow.truncated_normal([hidden_nodes, char_size], -0.1, 0.1))
            b = tensorflow.Variable(tensorflow.zeros([char_size]))
            logits = tensorflow.matmul(outputs_all_i, w) + b
            loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(logits, labels_all_i))
            # optimize4
            optimizer = tensorflow.train.GradientDescentOptimizer(10.).minimize(loss, global_step=global_step)
            # test
            #test_data = tensorflow.placeholder(tensorflow.float32, shape=[1, char_size])
            #test_output = tensorflow.Variable(tensorflow.zeros([1, hidden_nodes]))
            #test_state = tensorflow.Variable(tensorflow.zeros([1, hidden_nodes]))
            #reset_test_state = tensorflow.group(
                #test_output.assign(tensorflow.zeros([1, hidden_nodes])),
                #test_state.assign(tensorflow.zeros([1, hidden_nodes])))
            #test_output, test_state = self.lstm(test_data, test_output, test_state)
            #test_prediction = tensorflow.nn.softmax(tensorflow.matmul(test_output, w) + b)
        with tensorflow.Session(graph=graph) as sess:
            tensorflow.global_variables_initializer().run()
            offset = 0
            saver = tensorflow.train.Saver()
            for step in range(max_steps):
                offset = offset % len(X)
                if offset <= (len(X) - batch_size):
                    batch_data = X[offset: offset + batch_size]
                    batch_labels = y[offset: offset + batch_size]
                    offset += batch_size
                else:
                    to_add = batch_size - (len(X) - offset)
                    batch_data = numpy.concatenate((X[offset: len(X)], X[0: to_add]))
                    batch_labels = numpy.concatenate((y[offset: len(X)], y[0: to_add]))
                    offset = to_add
                _, training_loss = sess.run(
                    [optimizer, loss], feed_dict={data: batch_data, labels: batch_labels})
                if step % 10 == 0:
                    print('training loss at step %d: %.2f (%s)' % (
                        step, training_loss, datetime.datetime.now()))
                    if step % save_every == 0:
                        saver.save(sess, checkpoint_directory + '/model', global_step=step)
        test_start = 'I plan to make the world a better place '
        with tensorflow.Session(graph=graph) as sess:
            tensorflow.global_variables_initializer().run()
            model = tensorflow.train.latest_checkpoint(checkpoint_directory)
            saver = tensorflow.train.Saver()
            saver.restore(sess, model)
            reset_test_state.run() 
            test_generated = test_start
            for i in range(len(test_start) - 1):
                test_X = numpy.zeros((1, char_size))
                test_X[0, char2id[test_start[i]]] = 1.0
                _ = sess.run(test_prediction, feed_dict={test_data: test_X})
            test_X = numpy.zeros((1, char_size))
            test_X[0, char2id[test_start[-1]]] = 1.
            for i in range(500):
                prediction = test_prediction.eval({test_data: test_X})[0]
                next_char_one_hot = self.sample(prediction)
                next_char = id2char[numpy.argmax(next_char_one_hot)]
                test_generated += next_char
                test_X = next_char_one_hot.reshape((1, char_size))
            print(test_generated)

    def lstm(self, i, o, state):
        """
        LSTM Cell
        Given input, output, external state, it will return output and state.
        Output starts off empty, LSTM cell calculates it.
        """
        input_gate = tensorflow.sigmoid(tensorflow.matmul(i, w_ii) + tensorflow.matmul(o, w_io) + b_i)
        forget_gate = tensorflow.sigmoid(tensorflow.matmul(i, w_fi) + tensorflow.matmul(o, w_fo) + b_f)
        output_gate = tensorflow.sigmoid(tensorflow.matmul(i, w_oi) + tensorflow.matmul(o, w_oo) + b_o)
        memory_cell = tensorflow.sigmoid(tensorflow.matmul(i, w_ci) + tensorflow.matmul(o, w_co) + b_c)
        state = forget_gate * state + input_gate * memory_cell
        output = output_gate * tensorflow.tanh(state)
        return output, state

    def sample(self, prediction):
        """
        Given a probability of each character, return a likely character, one-hot encoded.
        The prediction will give us an array of probabilities of each character.
        We'll pick the most likely and one-hot encode it.
        """
        r = random.uniform(0, 1)
        s = 0
        char_id = len(prediction) - 1
        for i in range(len(prediction)):
            s += prediction[i]
            if s >= r:
                char_id = i
                break
        char_one_hot = numpy.zeros(shape=[char_size])
        char_one_hot[char_id] = 1.0
        return char_one_hot

if __name__ == '__main__':
    wiki_generator = WikiGenerator()
    wiki_generator()
