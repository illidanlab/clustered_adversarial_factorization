
import os
import scipy.io
from LoadData import Data
from Train import TrainModel
# define Frobenius norm

def main():
    maxiter = 20000
    du1 = 50
    du2 = 25
    D = 512
    samplesize = 64
    lambdad = 1e-2
    lambdar = 1
    lambdax = 1e-2
    lambdau = 1e-2
    lambdav = 1e-3
    step = 10
    pretrainstep = 30
    directory = "result"
    datapath = "datapath"
    pretrainstep = 30000
    lambdad = 1:
    D_1 = D
    D_2 = D
    D_3 = D
    data_loader = Data(datapath)
    data = data_loader.load_data()
    X_notmissing_data = data['X_notmissing_data']
   	X_missing_data = data['X_missing_data']
    weight = data['weight']
    index1 = data['index1']
    index2 = data['index2']
    X_full = data['X_full']
    missing_ID = np.isnan(X_missing_data)
    X_missing_data[missing_ID] = 0
    trainer = TrainModel()
    R_sample = trainer.train(X_missing_data, missing_ID, X_notmissing_data, samplesize, weight, index1, index2, iter+1,  D_1, D_2, D_3, du1, du2, lambdad, lambdar, lambdax, lambdau, lambdav, maxiter, step, pretrainstep)
    missing_float = missing_ID.astype(float)
    error = np.linalg.norm(np.multiply(missing_float, (R_sample - X_full)), 'fro') / np.linalg.norm(np.multiply(missing_float, X_full), 'fro')
if __name__ == "__main__":
    main()

