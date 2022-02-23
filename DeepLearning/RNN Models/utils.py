import collections
import re

src_tm = "http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt"

def download(url): # down load text file from a URL 
  import requests
  import os

  fpath = os.path.join("/content/", url.split('/')[-1])
  r = requests.get(url, stream=True, verify=True)
  with open(fpath, 'wb') as f:
      f.write(r.content)
  return fpath

def readtextonly(src_file): 
  import re
  PATTERN = "[^A-Za-z]+"
  with open(src_file,'r') as f:
    lines = f.readlines()
  return [re.sub(PATTERN,' ',line).strip().lower() for line in lines]

lines = readtextonly(download(src_tm))

def tokenize(lines,token = 'word'): # split text to tokens 
  if isinstance(lines, str):
    if token == 'word':return [lines.split()][0]
    elif token =='char':return [list(line) for line in lines]
  else:
    if token == 'word':return [line.split() for line in lines]
    elif token =='char':return [list(line) for line in lines]


def corpus_count(tokens):
  if isinstance(tokens[0],list):
    _tokens = [token for line in tokens for token in line]
    return collections.Counter(_tokens)


class Vocab:
  def __init__(this, tokens = None, min_freq = 0) -> None:
      if tokens is None: tokens = []  
      corpus = corpus_count(tokens) #get a dictionary of all tokens with there frequencies {token: frequency}
      this.sortedTokens = sorted(corpus.items(), key=lambda x:x[1], reverse=True) #sort the tokens from the heigh to low fraquency 
      this.index2token = ['<unk>'] # index for the unknown is 0
      this.token2index = { token:idx for idx,token in enumerate(this.index2token) } # dictionary of  tokens and there index based on the frequancy
      # create two list one of the indexes(token2index) and one for the tokens(index2token) based on the parametre min_freq
      for token, freq in this.sortedTokens:
        if freq < min_freq: break
        if token not in this.token2index:
          this.index2token.append(token)
          this.token2index[token] = len(this.index2token) - 1


  def __len__(this): #get the length of the list
    return len(this.index2token)
  
  def __getitem__(this,tokens): # get index of a token
    if not isinstance(tokens, (list, tuple)):
      return this.token2index.get(tokens,this.unk)
    return [this.__getitem__(item) for item in tokens]
  
  def to_tokens(this, indices): #get token of a index
    if not isinstance(indices, (list, tuple)):
      return this.index2token[indices]
    return [this.index2token[index] for index in indices]

  def unk(this):return 0

# tokens = tokenize(lines)
# corpus = corpus_count(tokens)
# vocab = Vocab(tokens)
# print(list(vocab.token2index.items())[:20])

def load_corpus(src, max_tokens=-1):
  lines = readtextonly(download(src))
  tokens = tokenize(lines, 'char')
  vocab = Vocab(tokens)
  corpus = [vocab[token] for line in tokens for token in line]
  if max_tokens > 0:
    corpus = corpus[:max_tokens]
  return corpus, vocab


corpus, vocab = load_corpus(src_tm)

