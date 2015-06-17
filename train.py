from keras.layers.core import *
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import *
from keras.objectives import *
from keras.constraints import *
from keras.regularizers import *
from keras.layers.recurrent import *
from keras.layers.advanced_activations import LeakyReLU

def get_model(num_word, num_label, emb_dim=50, hid_dim=[300], dropout=False, activation='tanh', reg=1e-3, preinit=None, truncate=7):
    word_emb = Embedding(num_word, emb_dim, W_constraint=unitnorm)
    model = Sequential()
    model.add(word_emb)
    model.add(GRU(emb_dim, hid_dim[0], truncate_gradient=truncate))
    n_in = n_out = hid_dim[0]
    for n_out in hid_dim[1:]:
        model.add(Dense(n_in, n_out, W_regularizer=l2(reg)))
        model.add(Activation('tanh'))
        if dropout:
            model.add(Dropout(0.5))
        n_in = n_out
    model.add(Dense(n_out, num_label))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    import json

    max_epoch = 100

    from dataset import Dataset
    dataset = Dataset()
    model = get_model(len(dataset.word_vocab), len(dataset.label_vocab), dropout=True, truncate=20)
    model.compile(optimizer='adagrad', loss='categorical_crossentropy')

    log = open('train.log', 'wb')

    for epoch in range(max_epoch):
        loss, acc, total = 0, 0, 0
        for x, y, in dataset.train:
            loss_, acc_ = model.train(x, y, accuracy=True)
            loss += loss_ * len(x)
            acc += acc_ * len(x)
            total += len(x)
        loss /= float(total)
        acc /= float(total)
        l = {'epoch': epoch, 'train_loss': loss, 'train_acc': acc}

        loss, acc, total = 0, 0, 0
        for x, y, in dataset.dev:
            loss_, acc_ = model.train(x, y, accuracy=True)
            loss += loss_ * len(x)
            acc += acc_ * len(x)
            total += len(x)
        loss /= float(total)
        acc /= float(total)
        l.update({'dev_loss': loss, 'dev_acc': acc})

        p = json.dumps(l, sort_keys=True)
        print p
        log.write(p + "\n")

