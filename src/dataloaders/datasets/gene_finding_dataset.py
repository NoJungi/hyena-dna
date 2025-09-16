from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
import h5py

from hg38_char_tokenizer import CharacterTokenizer


"""

Dataset for sampling intervals from human refernce genome.

"""

# augmentations

string_complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'a': 't', 'c': 'g', 'g': 'c', 't': 'a'}

def string_reverse_complement(seq):  # don't think I will need this...
    rev_comp = ''
    for base in seq[::-1]:
        if base in string_complement_map:
            rev_comp += string_complement_map[base]
        # if bp not complement map, use the same bp
        else:
            rev_comp += base
    return rev_comp


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
        batch_size=64,
        add_eos=False,
        last_chunk_overlap=False,
    ):
        """
        Initialize dataset used by BEND for gene finding task. All sequences longer than max_length are 
        split into chunks of max_length. For sequences that are not a multiple of max_length, the last 
        chunk is either overlapping with the previous chunk or padded to max_length. When the last chunk 
        is overlapping (last_chunk_overlap=True) with previous chunk the overlap is not accounted for in 
        the loss, since these positions are already predicted in the previous chunk. Overlapping allows 
        to have the same context length for each chunk. When last_chunk_overlap=False, the last chunk is 
        padded to max_length resulting in shorter context for the last chunk.
        
        Args:
            split:                  'train', 'val', 'test'
            bed_file:               path to .bed file with columns: chromosome, start, end, strand, length, split
            fasta_file:             path to .fasta file
            label_file:             path to .hdf5 file containing labels for each nucleotide
            max_length:             maximum length of sequences
            batch_size:             batch size
            add_eos:                whether to add end-of-sequence and start-of-sequence token
            last_chunk_overlap:     whether to overlap last chunk or just add padding
        """

        self.max_length = max_length
        self.add_eos = add_eos
        if self.add_eos:
            self.max_length -= 2 # account for adding eos and sos
        self.batch_size = batch_size
        self.last_chunk_overlap = last_chunk_overlap
        #self.seq_length = max_length
        #self.step = self.max_length

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
        i=0
        for row in df_raw.iterrows():
            print(f"i: {i}")
            print(f"len df: {len(self.df)}")

            label_index = row[0]
            row = row[1]

            print(f"label_index: {label_index}")
            print(f"row: {row}")

            if row['length'] <= self.max_length: # no chunks needed, only padding
                self.df.loc[i] = [row['chromosome'], row['start'], row['end'], row['strand'], row['length'], label_index, 0]
                i +=1
            else: # cut into chunks of max_length
                nr_full_chunks = int(row['length'] / self.max_length)
                last_chunk_length = row['length'] % self.max_length

                label_start = 0
                start = row['start']

                for _ in range(nr_full_chunks):
                    end = start + self.max_length
                    self.df.loc[i] = [row['chromosome'], start, end, row['strand'], self.max_length, label_index, label_start]
                    start += self.max_length
                    label_start += self.max_length
                    i += 1
                if last_chunk_length > 0:
                    if self.last_chunk_overlap: # set the starting index of nucleotides in the last chunk to be max_length before the end
                        start = row['end'] - self.max_length
                    self.df.loc[i] = [row['chromosome'], start, row['end'], row['strand'], last_chunk_length, label_index, label_start]
                    i += 1

        self.fasta = FastaInterval(fasta_file = fasta_file)

        # read label file:
        with h5py.File(label_path, "r") as f:
            self.labels = list(f['labels'])

    def __len__(self):
        return len(self.df)

    def replace_value(self, x, old_value, new_value):
        return torch.where(x == old_value, new_value, x)

    def __getitem__(self, idx):
        """Returns a sequence of specified len"""

        # sample the row from df
        row = self.df.iloc[idx]  
        # row = (chr, start, end, strand, length, label_index, label_start)
        chr_name, start, end, strand, length, label_index, label_start = (row[0], row[1], row[2], row[3], row[4], row[5], row[6])

        # get sequence
        seq = self.fasta(chr_name, start, end)

        # tokenize sequence (and add padding if sequence is shorter than max_length)
        seq = self.tokenizer(seq,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length",
            max_length=self.max_length, #default: add padding on left
            truncation=True,
        )
        seq = seq["input_ids"]  # get input_ids
        seq = torch.LongTensor(seq) # convert to tensor

        # get Target: classes for each nucleotide (and add padding labels for shorter sequences)
        label_index = row['label_index']
        print(f"label_index: {label_index}")
        print(f"row[5]: {row[5]}")
        label_start = row['label_start']
        print(f"label_start: {label_start}")
        print(f"row[6]: {row[6]}")
        length = row['length']
        print(f"length: {length}")
        print(f"row[4]: {row[4]}")

        label = self.labels[label_index]
        label = label[label_start:(label_start + length)]
        # add padding labels on the left if length < max_length
        if length < self.max_length:
            pad_length = self.max_length - length
            label = np.pad(label, (pad_length, 0), 'constant', constant_values=-100)  # -100 is the ignore index for CrossEntropyLoss
        label = torch.LongTensor(label)

        return seq, label

if __name__ == '__main__':

    fasta_file = '/home/s-nojung/jupyterhub/Masterarbeit/Code/hyena-dna/data/gene_finding/GRCh38.primary_assembly.genome.fa'
    bed_file = '/home/s-nojung/jupyterhub/Masterarbeit/Code/hyena-dna/data/gene_finding/gene_finding.bed'
    label_file = '/home/s-nojung/jupyterhub/Masterarbeit/Code/hyena-dna/data/gene_finding/gene_finding.hdf5'

    max_length = 1026

    dataset = BendDataset(split='test',
        bed_file=bed_file,
        fasta_file=fasta_file,
        label_file=label_file,
        max_length=max_length,
        add_eos=False)

    print("LENGTH DF:", dataset.__len__())

    seq, label = dataset.__getitem__(8)
    print(len(seq))
    print(seq)
    print(len(label))
    print(label)


