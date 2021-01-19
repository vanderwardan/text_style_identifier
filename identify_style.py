import sys
import torch, torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import string

class Model(nn.Module):
    def __init__(self, tokens_len):
        super(self.__class__, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(5, 20, kernel_size=5, padding=1)
        tmp = math.floor(((tokens_len - 2) / 2 - 3) / 2 + 1)
        self.batchnorm1 = nn.BatchNorm1d(5)
        self.dropout1 = nn.Dropout(0.5)
        
        self.dense1 = nn.Linear(20 * tmp, 1000)
        self.dense2 = nn.Linear(1000, 50)
        self.batchnorm2 = nn.BatchNorm1d(50)
        self.dropout2 = nn.Dropout(0.5)
        self.dense3 = nn.Linear(50, 3)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool1d(x, 2, 2)
        x = self.batchnorm1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = F.max_pool1d(x, 2, 2)
        
        x = x.view(-1, x.shape[1] * x.shape[2])
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.batchnorm2(x)
        x = self.dropout2(x)
        x = F.relu(self.dense3(x))
        
        return self.softmax(x)

def remove_punctuation(a):
    black_symbols = ''.join([chr(num) for num in range(32)]) + string.punctuation + '”“’…—'
    res = a.translate(str.maketrans('', '', black_symbols)).lower()
    res = res.replace('\n', ' ').replace('section', ' ').replace('chapter', ' ')
    return res

if __name__ == "__main__":
	try:
		model = torch.load('MODEL')
		model.eval()

		vectorizer = pickle.load(open("tfidf.pickle", 'rb'))

		x = remove_punctuation(sys.argv[1])
		x = vectorizer.transform([x])
		x = np.array(x.todense())
		x = torch.tensor(x[:, None, :], dtype=torch.float32)

		y = np.argmax(model(x).detach().numpy())

		if y == 0:
		    print('Разговорный стиль')
		elif y == 1:
		    print('Xудожественная литература')
		else:
		    print('Техническая литература')
	except:
		if len(sys.argv) == 1:
			print("No text is provided")
		else:
			print("Something's wrong")