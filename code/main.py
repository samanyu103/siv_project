import preprocessing
import data_loader
from textcnn import CNN_NLP
from torch.optim import Adam
from torch.utils.data import DataLoader
import pdb 
import torch
from colorama import Fore
from tqdm import tqdm
import matplotlib.pyplot as plt 
config = {

    'epochs': 50,
    'batch_size': 10
}

class Trainer:
    def __init__(self, root):

        #get the data
        self.train_data=data_loader.Dataloader(root)
        self.val_data=data_loader.Dataloader(root, mode = 'val')
        print(f"train data: {len(self.train_data)}")
        print(f"val data: {len(self.val_data)}")


        #the dataloader
        self.train_dataloader = DataLoader(self.train_data, 
                                            shuffle = True, 
                                            batch_size = config['batch_size'],
                                            collate_fn = self.collate)
        
        self.val_dataloader = DataLoader(self.val_data, 
                                            shuffle = False, 
                                            batch_size = config['batch_size'],
                                            collate_fn = self.collate)
        
        #do the preprocessing step
        self.preprocessor=preprocessing.Preprocessor(root, self.train_data.reviews)
        self.vocab_size = len(self.preprocessor.dic)

        #get the model
        self.model = CNN_NLP(vocab_size = self.vocab_size)

        #the optimizer
        self.optimizer = Adam(self.model.parameters(), lr = 0.0001)
    

    def collate(self, x):
        
        y = torch.tensor([i[1] for i in x])
        x = [torch.tensor(self.preprocessor.preprocess(i[0])) for i in x]
        x = torch.nn.utils.rnn.pad_sequence(x, batch_first = True)
        return x, y


    def loss(self, y, y_pred):
    
        y = torch.nn.functional.one_hot(y)
        loss = torch.log(y_pred)*y
        loss = loss.sum(-1).mean()
        loss = -1*loss
        return loss

    def train_epoch(self):
        losses = []
        pbar = tqdm(self.train_dataloader, ncols = 120, bar_format = "{l_bar}%s{bar:50}%s{r_bar}" % (Fore.BLUE, Fore.RESET))
        pbar.set_postfix({"loss":100})

        for (x,y) in pbar:
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.loss(y, y_pred)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        
            #update progess bar
            pbar.set_postfix({"loss":loss.item()})
            pbar.update()

        return (sum(losses)/len(losses))

    def validate(self):
        correct = 0
        total = 0
        with torch.no_grad():

            for (x,y) in self.val_dataloader:
              
                y_pred = self.model(x)
                y_pred = torch.argmax(y_pred, dim = -1)
                correct += (y_pred == y).long().sum()
                total += y.shape[0]

            return (correct/total).item()



    def train(self):
        losses = []
        accuracies = []
        for i in range(config['epochs']):
            loss = self.train_epoch()
            print()
            
            accuracy = self.validate()
            print(f"Epoch: {i}, Loss: {loss}, Accu: {accuracy}")

            losses.append(loss)
            accuracies.append(accuracy)
        
        plt.plot(losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("loss vs epoch")
        plt.savefig("train_loss.png")
        plt.close()

        plt.plot(accuracies)
        plt.xlabel("epoch")
        plt.ylabel("val accuracy")
        plt.title("val accuracy vs epoch")
        plt.savefig("val_accu.png")
        


trainer = Trainer('/home/cs5190446/Desktop/siv/op_spam_v1.4')
trainer.train()