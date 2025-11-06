
from torch.utils.data import Dataset
import numpy as np
from scipy.sparse import csr_matrix

class TrnData(Dataset):
    def __init__(self, csrmat):
        if isinstance(csrmat, csr_matrix):
            coomat = csrmat.tocoo()
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()

    def negSampling(self,pos,user):
        negs = np.zeros(len(pos)).astype(np.int32)
        for i,u in enumerate(user):
            while True:
                iNeg = np.random.randint(len(pos))
                if (u, pos[iNeg]) not in self.dokmat:
                    break
            negs[i] = iNeg
        return negs

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx]
class TstData(Dataset):
    def __init__(self, csrmat, trnMat):
        self.csrmat = (trnMat!= 0) * 1.0
        if isinstance(csrmat, csr_matrix):
            coomat = csrmat.tocoo()
        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
