import numpy
from sklearn.decomposition import PCA
import sys

class Knn:

    def __init__(self):
        self._cifar = dict()
        self._train = {'labels': list(), 'data':[]}
        self._gray_train = {'labels': list(), 'data':[]}
        self._test = {'labels': list(), 'data':[]}
        self._gray_test = {'labels': list(), 'data':[]}
        self._pca = PCA()
        self._result = list()

    def pickle2dic(self,file):
        import pickle
        pickle_file = open(file,"rb")
        dic = pickle.load(pickle_file, encoding='bytes')
        pickle_file.close()
        self._cifar = dic

    def eu_distance(self,x,y):
        import math
        distance = 0.0
        if len(x) != len(y):
            return False
        for i in range(len(x)-1):
            distance += pow(x[i]-y[i],2)
        distance = math.sqrt(distance)
        return distance

    def train(self, n, d):
        self._train['label'] = self._cifar[b'labels'][n:1000]
        self._train['data'] = self._cifar[b'data'][n:1000,:]
        self._test['label'] = self._cifar[b'labels'][0:n]
        self._test['data'] = self._cifar[b'data'][0:n,:]
        r, g, b = self._train['data'][:,0:1024], self._train['data'][:,1024:2048], self._train['data'][:,2048:3072]
        self._gray_train = dict(self._train)
        self._gray_train['data'] = 0.299*r + 0.587*g + 0.114*b
        self._pca = PCA(n_components = d, svd_solver='full')
        self._pca = self._pca.fit(self._gray_train['data'])
        self._gray_train['data'] = self._pca.transform(self._gray_train['data'])

    def knn(self, k):
        distance = list(list(tuple()))
        min_k_distance = list(list(tuple()))
        labels = list()
        r, g, b = self._test['data'][:, 0:1024], self._test['data'][:, 1024:2048], self._test['data'][:, 2048:3072]
        self._gray_test = dict(self._test)
        self._gray_test['data'] = 0.299 * r + 0.587 * g + 0.114 * b
        self._gray_test['data'] = self._pca.transform(self._gray_test['data'])
        for i in range(self._gray_test['data'].shape[0]):
            distance.append(list())
            for j in range(self._gray_train['data'].shape[0]):
                distance[i].append((self.eu_distance(self._gray_train['data'][j],self._gray_test['data'][i]),self._gray_train['label'][j]))
            distance[i].sort(key=lambda x: x[0])
            min_k_distance.append(list())
            min_k_distance[i] = distance[i][:k]
            vote = [0] * 10
            for l in range(len(min_k_distance[i])):
                voting_weight = 1 / pow(min_k_distance[i][l][0],2)
                vote[min_k_distance[i][l][1]] += voting_weight
            max_vote = max(vote)
            labels.append(vote.index(max_vote))
        self._result = list(labels)

    def write2file(self,filename):
        file = open(filename,"w")
        for index, value in enumerate(self._result):
            file.write('{} {}\n'.format(value, self._test['label'][index]))
        file.close()

def main():
    if len(sys.argv) != 5:
        print(len(sys.argv))
        print(sys.argv)
        print("Invalid number of arguments")
        return False
    args = sys.argv
    k, d, n, file = int(args[1]), int(args[2]), int(args[3]), args[4]
    knn = Knn()
    knn.pickle2dic(file)
    knn.train(n,d)
    knn.knn(k)
    knn.write2file("./output.txt")

if __name__ == "__main__":
    main()
