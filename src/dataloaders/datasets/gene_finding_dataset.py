from pathlib import Path
from pyfaidx import Fasta
import pandas as pd
import torch
import numpy as np
import h5py

from src.dataloaders.datasets.hg38_char_tokenizer import CharacterTokenizer


"""

Dataset for sampling intervals from human refernce genome.

"""

class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))     

    def __call__(self, chr_name, start, end):
        """
        Get DNA Sequence from coordinates
        """
        chromosome = self.seqs[chr_name]
        seq = str(chromosome[start:end])

        return seq

class BendDataset(torch.utils.data.Dataset):

    '''
    Loop thru bed file, retrieve (chr, start, end), query fasta file for sequence.
    
    '''

    def __init__(
        self,
        split,
        bed_file,
        fasta_file,
        label_file,
        max_length,
        add_eos=False,
        last_chunk_overlap=False,
        pad_value = -100
    ):
        """
        Initialize dataset used by BEND for gene finding task. All sequences longer than max_length are 
        split into chunks of max_length. For sequences that are not a multiple of max_length, the last 
        chunk is either overlapping with the previous chunk or padded to max_length. When the last chunk 
        is overlapping (last_chunk_overlap=True) with previous chunk the overlap is not accounted for in 
        the loss of validation and test set, since these positions are already predicted in the previous 
        chunk. Overlapping allows to have full context length for each chunk. When last_chunk_overlap=False, 
        the last chunk is padded to max_length resulting in shorter context for the last chunk.
        
        Args:
            split:                  'train', 'valid', 'test'
            bed_file:               path to .bed file with columns: chromosome, start, end, strand, length, split
            fasta_file:             path to .fasta file
            label_file:             path to .hdf5 file containing labels for each nucleotide
            max_length:             maximum length of sequences
            add_eos:                whether to add end-of-sequence and start-of-sequence token
            last_chunk_overlap:     whether to overlap last chunk or just add padding
            pad_value:          labels that marked as padding can be ignored for loss and other metrices
        """

        self.max_length = max_length
        self.add_eos = add_eos
        if self.add_eos:
            self.max_length -= 2 # account for adding eos and sos
        self.last_chunk_overlap = last_chunk_overlap
        self.pad_value = pad_value

        self.tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
                model_max_length=self.max_length,
                add_special_tokens=True if self.add_eos else False,
                truncation=True
        )

        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'

        label_path = Path(label_file)
        assert label_path.exists(), 'path to .hdf5 file must exist'

        # read bed file
        df_raw = pd.read_csv(
            str(bed_path), 
            sep='\t', 
            usecols=['chromosome', 'start', 'end','strand', 'length', 'split']
        )
        df_raw = df_raw.reset_index()
        df_raw = df_raw[df_raw['split'] == split] # select split

        # get intervals with length max_length
        self.df = pd.DataFrame(columns=['chromosome', 'start', 'end','strand', 'length', 'label_index', 'label_start'])
        for row in df_raw.iterrows():
            label_index = row[0]
            row = row[1]

            if row['length'] <= self.max_length: # no chunks needed, only padding is added later in __get_item__()
                self.df.loc[len(self.df)] = [row['chromosome'], row['start'], row['end'], row['strand'], row['length'], label_index, 0]
            else: # cut into chunks of max_length
                nr_full_chunks = int(row['length'] / self.max_length)
                last_chunk_length = row['length'] % self.max_length

                label_start = 0
                seq_start = row['start']

                for _ in range(nr_full_chunks):
                    seq_end = seq_start + self.max_length
                    self.df.loc[len(self.df)] = [row['chromosome'], seq_start, seq_end, row['strand'], self.max_length, label_index, label_start]
                    seq_start += self.max_length
                    label_start += self.max_length

                if last_chunk_length > 0:
                    if self.last_chunk_overlap: # set the starting index of sequence so that it contains the last max_length nucleotides
                        seq_start = row['end'] - self.max_length
                        if split == 'train': # no padding for training set, use overlap 2 in loss both times
                            label_start = row['length'] - self.max_length
                    self.df.loc[len(self.df)] = [row['chromosome'], seq_start, row['end'], row['strand'], last_chunk_length, label_index, label_start]

        self.fasta = FastaInterval(fasta_file = fasta_file)

        # read label file:
        with h5py.File(label_path, "r") as f:
            self.labels = list(f['labels'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""

        # sample the row from df
        row = self.df.iloc[idx]  
        # row = (chr, seq_start, seq_end, strand, length, label_index, label_start)
        chr_name, seq_start, seq_end, strand, length, label_index, label_start = (row[0], row[1], row[2], row[3], row[4], row[5], row[6])
        print(strand)
        # get sequence
        seq = self.fasta(chr_name, seq_start, seq_end)

        # tokenize sequence (and add padding if sequence is shorter than max_length)
        seq = self.tokenizer(seq,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos not params in __init__
            padding="max_length",
            max_length=self.max_length, #default: add padding on left
            truncation=True,
        )
        seq = seq["input_ids"]          # get input_ids
        seq = torch.LongTensor(seq)     # convert to tensor

        # get Target: classes for each nucleotide (and add padding labels for shorter sequences)
        label = self.labels[label_index]
        label = label[label_start:(label_start + length)]

        # add padding labels on the left if length < max_length
        if length < self.max_length:
            pad_length = self.max_length - length
            label = np.pad(label, (pad_length, 0), 'constant', constant_values=self.pad_value)  # default: -100 is the ignore index for CrossEntropyLoss for HyenaDNA
        label = torch.LongTensor(label)

        return seq, label

if __name__ == '__main__':

    import os
    from pathlib import Path

    base_dir = Path(__file__).parent
    data_dir = os.path.join(base_dir, "../../../data")
    fasta_file = os.path.join(data_dir, 'gene_finding/GRCh38.primary_assembly.genome.fa')
    bed_file = os.path.join(data_dir, 'gene_finding/gene_finding.bed')
    label_file = os.path.join(data_dir, 'gene_finding/gene_finding.hdf5')

    max_length = 1026

    dataset_train = BendDataset(split='train',
        bed_file=bed_file,
        fasta_file=fasta_file,
        label_file=label_file,
        max_length=max_length,
        add_eos=False)

    dataset_val = BendDataset(split='valid',
        bed_file=bed_file,
        fasta_file=fasta_file,
        label_file=label_file,
        max_length=max_length,
        add_eos=False)

    dataset_test = BendDataset(split='test',
        bed_file=bed_file,
        fasta_file=fasta_file,
        label_file=label_file,
        max_length=max_length,
        add_eos=False)
    print(dataset_train.__len__())
    print(dataset_val.__len__())
    print(dataset_test.__len__())


