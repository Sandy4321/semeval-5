#!/usr/bin/env python
""" convert.py
Usage:
    convert.py [--force]
"""

__author__ = 'victor'

import os
import logging
import codecs
from spacy.en import English
from collections import OrderedDict
from stanza.text.vocab import SennaVocab, Vocab
import json
import cPickle as pkl
from docopt import docopt

def get_dep_parse(words, e1_index, e2_index, nlp):
    words = nlp(' '.join(words))
    e1 = OrderedDict()
    e = words[e1_index]
    while e != e.head and e not in e1:
        e1[e] = e.dep_
        e = e.head
    e1[e] = e.dep_
    e2 = OrderedDict()
    e = words[e2_index]
    while e != e.head and e not in e1 and e not in e2:
        e2[e] = e.dep_
        e = e.head
    e2[e] = e.dep_
    parse1 = []
    for tok, edge in e1.items():
        parse1 += [tok.text.strip(), edge]
    parse2 = []
    for tok, edge in e2.items():
        parse2 += [tok.text.strip(), edge]
    return parse1, parse2


def parse_sent(line, nlp):
    id, sent = line.strip("\n\r").split("\t")
    mod = sent.replace('<e1>', ' SYM_E1_START ').replace('<e2>', ' SYM_E2_START ')
    mod = mod.replace('</e1>', ' SYM_E1_END ').replace('</e2>', ' SYM_E2_END ')
    words = nlp(mod.replace('  ', ' ').strip()).text.split()
    if 'SYM_E1_START' not in words or 'SYM_E2_START' not in words:
        raise Exception("Obtained bad sentence:\n{}{}".format(line, mod))
    e1_index = words.index('SYM_E1_START')
    e2_index = words.index('SYM_E2_START')
    if e1_index < e2_index:
        e2_index -= 2
    else:
        e1_index -= 2
    words = [w for w in words if 'SYM_' not in w]
    e1, e2 = get_dep_parse(words, e1_index, e2_index)
    return e1, e2


def parse_label(line):
    return line.strip("\n\r")


def parse_file(fname, nlp):
    """
    Example
8001	"The most common <e1>audits</e1> were about <e2>waste</e2> and recycling."
Message-Topic(e1,e2)
Comment: Assuming an audit = an audit document.
    """
    with codecs.open(fname, encoding='utf-8') as f:
        parses = []
        labels = []
        for i, line in enumerate(f):
            if i % 4 == 0:
                # sentence
                parses += [parse_sent(line, nlp)]
            elif i % 4 == 1:
                labels += [parse_label(line)]
        return parses, labels


def numericalize(split, word_vocab, rel_vocab, add=False):
    parses, labels = split
    assert len(parses) == len(labels)
    num_examples = len(labels)
    proc_parses = []
    proc_labels = []
    for i in xrange(num_examples):
        p1, p2 = parses[i]
        proc_parses.append(
            (word_vocab.words2indices([t.lower() for t in p1], add=add),
             word_vocab.words2indices([t.lower() for t in p2], add=add))
        )
        proc_labels.append(rel_vocab.add(labels[i]))
    return proc_parses, proc_labels


if __name__ == '__main__':
    args = docopt(__doc__)
    mydir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(mydir, 'SemEval2010_task8_training', 'TRAIN_FILE.TXT')
    test_file = os.path.join(mydir, 'SemEval2010_task8_testing_keys', 'TEST_FILE_FULL.TXT')
    logging.basicConfig(level=logging.INFO)

    logging.info('starting preprocessing')
    if os.path.isfile('train.json') and os.path.isfile('test.json') and not args['--force']:
        logging.info('train.json and test.json already exists. Skipping proprocessing.')
    else:
        nlp = English()
        with open('train.json', 'wb') as f:
            json.dump(parse_file(train_file, nlp), f, indent=2)
        with open('test.json', 'wb') as f:
            json.dump(parse_file(test_file, nlp), f, indent=2)

    logging.info('starting numericalization')
    word_vocab = SennaVocab()
    rel_vocab = Vocab()

    with open('train.json') as f:
        train = json.load(f)
    with open('test.json') as f:
        test = json.load(f)

    numericalize(train, word_vocab, rel_vocab, add=True)
    word_vocab = word_vocab.prune_rares(cutoff=2)
    word_vocab = word_vocab.sort_by_decreasing_count()
    rel_vocab = rel_vocab.sort_by_decreasing_count()
    train = numericalize(train, word_vocab, rel_vocab, add=False)
    test = numericalize(test, word_vocab, rel_vocab, add=False)

    with open('vocab.pkl', 'wb') as f:
        pkl.dump({'word': word_vocab, 'rel': rel_vocab}, f)

    with open('trainXY.json', 'wb') as f:
        json.dump(train, f, indent=2)

    with open('testXY.json', 'wb') as f:
        json.dump(test, f, indent=2)
