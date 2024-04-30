import os
from tqdm import tqdm 
class Dataloader:
    def __init__(self, root, mode = 'train'):
        self.reviews=[]
        i=0
        folders=['negative_polarity/deceptive_from_MTurk', 'negative_polarity/truthful_from_Web', 'positive_polarity/deceptive_from_MTurk', 'positive_polarity/truthful_from_TripAdvisor']
        if(mode == 'train'):
            subfolders=['fold'+str(i) for i in range(1,5)]
        else:
            subfolders=['fold'+str(i) for i in range(5,6)]
            
        print("Loading data !!")
        
        for folder in tqdm(folders):
            for subfolder in subfolders:
                path=root+'/'+folder+'/'+subfolder
                # print(path)

                files = os.listdir(path)

                # Loop through each file and read its contents
                for file_name in files:
                    file_path = os.path.join(path, file_name)
                    with open(file_path, 'r') as file:
                        content = file.read()
                        # print(content)
                        if(folder[18]=='d'):
                            deceptive=1
                        else:
                            deceptive=0
                        # print(deceptive)
                        self.reviews.append((content, deceptive))
                        i+=1
        # print(self.reviews)
    def __getitem__(self, i):
        return self.reviews[i]

    def __len__(self):
        return len(self.reviews)



# data=Dataloader('op_spam_v1.4')
# print(data.geti(16))