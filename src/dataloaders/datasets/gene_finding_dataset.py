from pathlib import Path
from pyfaidx import Fasta
import polars as pl
import pandas as pd
import torch
from random import randrange, random
import numpy as np
import h5py
import sys

sys.path.append("/home/s-nojung/jupyterhub/Masterarbeit/Code/hyena-dna")

from standalone_hyenadna import CharacterTokenizer


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
        overlap=False,
        tokenizer=None,
        #tokenizer_name=None,
        add_eos=False,
        replace_N_token=False,  # replace N token with pad token
        batch_size=32
    ):

        self.max_length = max_length
        #self.tokenizer_name = tokenizer_name
        self.tokenizer = tokenizer
        self.add_eos = add_eos
        self.replace_N_token = replace_N_token  
        self.batch_size = batch_size
        self.seq_length = max_length
        self.overlap = overlap
        if self.overlap:
            self.step = int(self.max_length/2)
        else:
            self.step = self.max_length


        bed_path = Path(bed_file)
        assert bed_path.exists(), 'path to .bed file must exist'

        label_path = Path(label_file)
        assert label_path.exists(), 'path to .hdf5 file must exist'

        # read bed file
        df_raw = pd.read_csv(str(bed_path), sep='\t', 
            usecols=['chromosome', 'start', 'end','strand', 'length', 'split']
        )
        df_raw = df_raw.reset_index()
        # select only split df
        df_raw = df_raw[df_raw['split'] == split]

        # get intervals with length max_length
        self.df = pd.DataFrame(columns=['chromosome', 'start', 'end','strand', 'length', 'label_index', 'label_start'])
        i=0
        for row in df_raw.iterrows():
            label_index = row[0]
            row = row[1]
            if row['length'] <= self.seq_length :  # -2 specialTokens
                self.df.loc[i] = [row['chromosome'], row['start'], row['end'], row['strand'], row['length'], label_index, 0]
                i +=1
            else:
                nr_full_chunks = int(row['length'] / self.seq_length)
                last_chunk_length = row['length'] % self.seq_length
                if overlap:
                    if (last_chunk_length - (self.max_length/2)) > self.max_length/5:
                        nr_full_chunks += nr_full_chunks
                    else:
                        nr_full_chunks += nr_full_chunks -1

                label_start = 0
                start = row['start']

                for j in range(0,nr_full_chunks):
                    end = start + self.seq_length
                    self.df.loc[i] = [row['chromosome'], start, end, row['strand'], self.seq_length, label_index, label_start]
                    start += self.step 
                    label_start += self.step
                    i += 1
                if last_chunk_length > 0:
                    label_start = row['length'] - self.seq_length
                    start = row['end'] - self.seq_length
                    self.df.loc[i] = [row['chromosome'], start, row['end'], row['strand'], self.seq_length, label_index, label_start]
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
        # sample a random row from df
        row = self.df.iloc[idx]
        # row = (chr, start, end, strand, length, label_index, label_start)
        chr_name, start, end, strand, length, label_index = (row[0], row[1], row[2], row[3], row[4], row[5])

        seq = self.fasta(chr_name, start, end)

        #if self.tokenizer_name == 'char':

        seq = self.tokenizer(seq,
            add_special_tokens=True if self.add_eos else False,  # this is what controls adding eos
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        seq = seq["input_ids"]  # get input_ids
        
        # convert to tensor
        seq = torch.LongTensor(seq)  # hack, remove the initial cls tokens for now

        if self.replace_N_token:
            # replace N token with a pad token, so we can ignore it in the loss
            seq = self.replace_value(seq, self.tokenizer._vocab_str_to_int['N'], self.tokenizer.pad_token_id)

        # get Target: classes for each nucleotide
        label_index = row['label_index']
        label_start = row['label_start']
        length = row['length']
        label = self.labels[label_index]
        label = label[label_start:(label_start + length)]
        label = torch.LongTensor(label)

        return seq, label

if __name__ == '__main__':

    fasta_file = '/home/s-nojung/jupyterhub/Masterarbeit/Code/hyena-dna/data/gene_finding/GRCh38.primary_assembly.genome.fa'
    bed_file = '/home/s-nojung/jupyterhub/Masterarbeit/Code/hyena-dna/data/gene_finding/gene_finding.bed'
    label_file = '/home/s-nojung/jupyterhub/Masterarbeit/Code/hyena-dna/data/gene_finding/gene_finding.hdf5'

    max_length = 1026

    tokenizer = CharacterTokenizer(
                characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
                model_max_length=max_length,
                add_special_tokens=False,
                padding_side='right')

    dataset = BendDataset(split='test',
        bed_file=bed_file,
        fasta_file=fasta_file,
        label_file=label_file,
        max_length=max_length,
        tokenizer=tokenizer,
        add_eos=False)

    print("LENGTH DF:", dataset.__len__())

    seq, label = dataset.__getitem__(8)
    print(len(seq))
    print(seq)
    print(len(label))
    print(label)
    print(strand)
    print(label_index)


