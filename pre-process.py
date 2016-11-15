import numpy as np
import h5py
import re
import sys
import operator
import argparse
import pickle

def split_caption(line):

  clean_line = clean_str(line.strip())
  words = clean_line.split(' ')
  # words = words[1:]

  return words

def get_idx(file_name):
  max_sent_len = 0
  word_to_idx = {}
  # Starts at 2 for padding
  with open(file_name, 'rb') as handle:
    word_to_idx = pickle.loads(handle.read())
  return word_to_idx


def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
  string = re.sub(r"\'s", " \'s", string) 
  string = re.sub(r"\'ve", " \'ve", string) 
  string = re.sub(r"n\'t", " n\'t", string) 
  string = re.sub(r"\'re", " \'re", string) 
  string = re.sub(r"\'d", " \'d", string) 
  string = re.sub(r"\'ll", " \'ll", string) 
  string = re.sub(r",", " , ", string) 
  string = re.sub(r"!", " ! ", string) 
  string = re.sub(r"\(", " ( ", string) 
  string = re.sub(r"\)", " ) ", string) 
  string = re.sub(r"\?", " ? ", string) 
  string = re.sub(r"\s{2,}", " ", string)    
  return string.strip().lower()


def main():
  global args
  parser = argparse.ArgumentParser(
      description =__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument('w2idx', help="word2idx file", type=str)
  parser.add_argument('caption', help="single caption to evaluate", type=str)
  parser.add_argument('--padding', help="padding around each sentence", type=int, default=4)
  parser.add_argument('--max_sent_len', help="max lengh of sentences in the training and testing set", type=str, default=30)
  args = parser.parse_args()



  # # Load data
  padding = args.padding
  max_sent_len = args.max_sent_len
  word_to_idx = get_idx(args.w2idx)
  words = split_caption(args.caption)
  sent = [word_to_idx[word] for word in words]
  # start padding
  sent = [1]*padding + sent
  # print sent
  # end padding
  if len(sent) < max_sent_len:
      sent.extend([1] * (max_sent_len - len(sent)))
  vec = " ".join(str(x) for x in sent)
  sys.exit(vec)


if __name__ == '__main__':
  main()
