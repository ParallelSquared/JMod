#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Source Code Form is subject to the terms of the Oxford Nanopore
Technologies, Ltd. Public License, v. 1.0.  Full licence can be found
at https://github.com/ParallelSquared/JMod/blob/main/LICENSE.txt
"""
"""
Created on Mon May 13 11:18:05 2024

@author: kevinmcdonnell
"""
from pyteomics import fasta, parser
import tqdm
import re

fasta_files = [
    "/Volumes/One Touch/PhD/Vostro/Python/Human/humanSwissProt.fasta", ## human
    "/Volumes/One Touch/PhD/Vostro/Python/ecoliDB/uniprot-proteome_UP000000625.fasta", ## ecoli
    "/Volumes/One Touch/PhD/Vostro/Python/Yeast/uniprot-proteome_UP000002311+reviewed_yes.fasta" ## yeast
    ]
fasta_files = [
    "/Users/kevinmcdonnell/Programming/Data/FASTA/humanSwissProt.fasta", ## human
    "/Users/kevinmcdonnell/Programming/Data/FASTA/uniprot-proteome_UP000000625.fasta", ## ecoli
    "/Users/kevinmcdonnell/Programming/Data/FASTA/uniprot-proteome_UP000002311+reviewed_yes.fasta" ## yeast
    ]
rule = 'Trypsin'
max_num_missed_cleavage = 2

all_fasta_seqs = []
all_protein_names=[]
for fasta_file in fasta_files:
    fasta_seqs = set()
    proteins =set()
    pep_to_prot = {}
    prot_num=0
    with fasta.read(fasta_file) as db:
         for entry in db:
            # break
            prot_num+=1
            proteins.update([re.search("[a-z]{2}\|(.*)?\|",entry.description)[1]])
            peps = parser.cleave(entry.sequence,rule=rule,
                                 missed_cleavages=max_num_missed_cleavage,
                                 min_length=7,
                                 max_length=45)
            # peps = {re.sub("I","L",i) for i in peps}
            fasta_seqs.update(peps)
            
    all_fasta_seqs.append(fasta_seqs)
    all_protein_names.append(proteins)

    
## remove seqs that are in multiple orgs
for i in range(len(all_fasta_seqs)):
    for j in range(len(all_fasta_seqs)):
        if i!=j:
            shared = all_fasta_seqs[i].intersection(all_fasta_seqs[j])
            print(len(shared))
            all_fasta_seqs[i] = all_fasta_seqs[i] - shared
            all_fasta_seqs[j] = all_fasta_seqs[j] - shared
            
            
# fasta_file = "/Volumes/Lab/fasta/JD_minusEcoli.fasta"

# rule = 'Trypsin'
# max_num_missed_cleavage = 2

# org_indices = ['Homo sapiens',"","Saccharomyces cerevisiae"]
# all_fasta_seqs = [set(),set(),set()]
# all_protein_names=[]
# # for fasta_file in fasta_files:
# fasta_seqs = set()
# proteins =set()
# pep_to_prot = {}
# prot_num=0
# with fasta.read(fasta_file) as db:
#      for entry in db:
#         # break
#         org = re.search("OS=([A-z]+ [A-z]+)",entry.description)[1]
#         idx = org_indices.index(org)
#         prot_num+=1
#         proteins.update([re.search("[a-z]{2}\|(.*)?\|",entry.description)[1]])
#         peps = parser.cleave(entry.sequence,rule=rule,
#                              missed_cleavages=max_num_missed_cleavage,
#                              min_length=7,
#                              max_length=45)
#         all_fasta_seqs[idx].update(peps)

# # all_fasta_seqs.append(fasta_seqs)
# all_protein_names.append(proteins)


##### create pep to protein dictionary

# # pep_to_prot = {}

# for fasta_file in fasta_files[:1]:
#     fasta_seqs = set()
#     proteins =set()
#     pep_to_prot = {}
#     prot_num=0
#     with fasta.read(fasta_file) as db:
#           for entry in db:
#             # break
#             prot_num+=1
#             protein= re.search("[a-z]{2}\|(.*)?\|",entry.description)[1]
#             peps = parser.cleave(entry.sequence,rule=rule,
#                                   missed_cleavages=max_num_missed_cleavage,
#                                   min_length=5,
#                                   max_length=45)
#             # peps = {re.sub("I","L",i) for i in peps}
            
#             for pep in peps:
#                 pep_to_prot.setdefault(pep,"")
#                 pep_to_prot[pep]+=protein+";"
            
            
            