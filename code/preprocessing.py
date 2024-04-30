import os
from tqdm import tqdm

class Preprocessor:
    
    def __init__(self, root, reviews = None):
        
        print("Intializing preprocessor !!")
        reviews=reviews
        
        # all (review, deceptive) tuples
        self.PAD_ID = 0
        self.UNK_ID = 1

        self.dic={'<pad>': 0, '<unk>':1}
        
        i=0
        for review in reviews:
            tokens=self.tokenize(review[0])
            for token in tokens:
                if (token in self.dic):
                    continue
                self.dic[token]=i
                i+=1
        # print(self.dic)

        
    def tokenize(self, sentence):
        return sentence.lower().split(' ')
    
    def preprocess(self, x):

        tokens = self.tokenize(x)
        token_id = []
        for i in tokens:
            if(i in self.dic):
                token_id.append(self.dic[i])
            else:
                token_id.append(self.dic['<unk>'])
        return token_id

        
# p=Preprocessor('op_spam_v1.4')