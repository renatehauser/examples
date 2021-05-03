###############################################################################
# Language Modeling on a subset of OMCS
#
# This file generates new sentences sampled from the language model
#
###############################################################################

import argparse

import torch

import data

import sys

parser = argparse.ArgumentParser(description='PyTorch OMCS Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='data/commonsense',
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str, default='models/model_40perc.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=200,
                    help='reporting interval')
# My addition:
parser.add_argument('--input', type=str, help='Give an input of one or several words delimited by white spaces',
                    required=False)
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda" if args.cuda else "cpu")

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f).to(device)
model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

is_transformer_model = hasattr(model, 'model_type') and model.model_type == 'Transformer'
if not is_transformer_model:
    hidden = model.init_hidden(1)

# My addition:
# if input is given, convert list of input words to list of corresponing indexes
if args.input is not None:
    input_words = args.input.split()
    input_idxs = []
    for word in input_words:
        try:
            idx = corpus.dictionary.word2idx[word]
        # if word is not in the vocabulary, exit the program and inform user
        except KeyError:
            sys.exit(f"'{word}' is not in the vocabulary. Please enter another word as input.")
        else:
            idx = torch.tensor(idx)
            idx_tensor = torch.reshape(idx, (1, 1))
            input_idxs.append(idx_tensor)
else:
    input_idxs = [torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)]
    #print(corpus.dictionary.idx2word[input])
    #print(input.shape)

with open(args.outf, 'w') as outf:
    with torch.no_grad():  # no tracking history
        # My Addition:
        i = -1
        for input in input_idxs:
            i += 1
            word = corpus.dictionary.idx2word[input]
            outf.write(word + ('\n' if i % 20 == 19 else ' '))
            output, hidden = model(input, hidden)
        word_weights = output.squeeze().div(args.temperature).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        # print("word_idx: ", word_idx)
        input.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        for i in range(len(input_idxs), args.words-1):
            if is_transformer_model:
                output = model(input, False)
                word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                input = torch.cat([input, word_tensor], 0)
            else:
                output, hidden = model(input, hidden)
                word_weights = output.squeeze().div(args.temperature).exp().cpu()
                word_idx = torch.multinomial(word_weights, 1)[0]
                #print("word_idx: ", word_idx)
                input.fill_(word_idx)
                #print("input: ", input)
                #print("input.shape: ", input.shape)

            word = corpus.dictionary.idx2word[word_idx]

            outf.write(word + ('\n' if i % 20 == 19 else ' '))

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))

