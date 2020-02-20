import numpy as np
import torch
import torch.utils.data
from MY_BERT import Constants


def paired_collate_fn(insts):

    src, tgt = list(zip(*insts))
    batch_seq,tgt_seq = collate_fn(src,tgt)

    assert batch_seq.size() == tgt_seq.size()
    return (batch_seq, tgt_seq)

def collate_fn(insts,tgt):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([inst + [Constants.PAD_id] * (max_len - len(inst)) for inst in insts]) #加入PAD
    tgt_seq = np.array([tag_label + [Constants.OTHER_TAG_id] * (max_len - len(tag_label)) for tag_label in tgt])

    batch_seq = torch.LongTensor(batch_seq)
    tgt_seq = torch.LongTensor(tgt_seq)

    return batch_seq,tgt_seq

class ClassficationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        src_insts, tgt_insts):

        assert src_insts
        assert not tgt_insts or (len(src_insts) == len(tgt_insts))


        self._src_insts = src_insts

        self._tgt_insts = tgt_insts

    @property
    def n_insts(self):
        ''' Property for dataset size '''
        return len(self._src_insts)

    def __len__(self):
        return self.n_insts

    def __getitem__(self, idx):
        if self._tgt_insts:
            return self._src_insts[idx], self._tgt_insts[idx]
        return self._src_insts[idx]
