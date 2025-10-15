import pandas as pd
import numpy as np
import re
import glob
import pmap
import os
from Bio import SeqIO
from bibis.sampling.gc import SetGCSampler
from bibis.seq.seqentry import SeqEntry
from libis.dinucl_shuffle import shuffle_seq_dinucl

import numpy as np
from pmap import pmap
GENOME = './genome/hg38.fa'

# HTS block
def A2G_read(path : str) -> pd.DataFrame:
    if not os.path.isdir(path):
        raise NameError('The presented path must be a directory')
        
    else:
        lst_paths = glob.glob(f'{path}*')
        lst_seq = []      
        lst_repl = []
        lst_cycles = []
        lst_hashes = []
        for p in lst_paths:
            temp_lst = [str(record.seq) for record in SeqIO.parse(p, format='fastq')]    
            replic = int(re.findall(string=p, pattern='_R(\d)_')[0])
            cycle = int(re.findall(string=p, pattern='_C(\d)_')[0])

            lst_seq.extend(temp_lst)
            lst_hashes.extend([hash(seq) for seq in temp_lst])
            lst_repl.extend([replic]*len(temp_lst))
            lst_cycles.extend([cycle]*len(temp_lst))
        pos_data = pd.DataFrame({'seq':lst_seq, 
                                    'label':np.ones(len(lst_seq), dtype= 'int'), 
                                    'cycle':lst_cycles, 'replic':lst_repl,
                                    'hash':lst_hashes})
        pos_data.sort_values(by = ['cycle'], inplace=True)
        pos_data = pos_data.drop_duplicates(subset=['hash'], keep='last').drop(['hash'],axis = 1)
        replic_sizes = pos_data.replic.value_counts()
              
    return pos_data




def A2GShuffleSeq(pos_data: pd.DataFrame, num_workers: int, 
                  num_negatives:int = 1, shuffle_type='mono') -> pd.DataFrame:
    '''
    Generation of negative controls with help of dinucleotide shuffle
    '''
    lst_seq = pos_data.seq.values
    lst_replics = np.concatenate([pos_data.replic.values for i in range(num_negatives)]) 
    lst_neg_seq = []

    if shuffle_type == 'dinucl':
        for i in range(num_negatives):
            temp_lst = list(pmap(f=lambda x: shuffle_seq_dinucl(x), seq= lst_seq, threads=num_workers))
            lst_neg_seq.extend(temp_lst)
            
    elif shuffle_type == 'mono':
        for i in range(num_negatives):
            temp_lst = list(pmap(f=lambda x: shuffle_seq_dinucl(x), seq= lst_seq, threads=num_workers))
            lst_neg_seq.extend(temp_lst)

    neg_data = pd.DataFrame({'seq':lst_neg_seq, 'label':np.zeros(len(lst_neg_seq), dtype = 'int'), 
                             'cycle':np.zeros(len(lst_neg_seq), dtype = 'int'), 'replic':lst_replics })
    neg_data['hash'] = [hash(seq) for seq in neg_data.seq]
    neg_data = neg_data.drop_duplicates(subset=['hash'], keep='last').drop(['hash'],axis = 1)
    print('Dinucleotide negatives were generated!')
    return neg_data

def MononuclShuffle(pos_data: pd.DataFrame, num_workers: int, num_negatives:int = 1, seed:int = 29) -> pd.DataFrame:
    '''
    Generation of negative controls with help of shuffle
    '''
    np.random.seed(seed)
    lst_seq = pos_data.seq.values
    lst_replics = np.concatenate([pos_data.replic.values for i in range(num_negatives)]) 
    lst_neg_seq = []

    for i in range(num_negatives):
        temp_lst = list(pmap(f=lambda x: ''.join(np.random.permutation(list(x))), seq= lst_seq, threads=num_workers))
        lst_neg_seq.extend(temp_lst)
    neg_data = pd.DataFrame({'seq':lst_neg_seq, 'label':np.zeros(len(lst_neg_seq), dtype = 'int'), 
                             'cycle':np.zeros(len(lst_neg_seq), dtype = 'int'), 'replic':lst_replics })
    neg_data['hash'] = [hash(seq) for seq in neg_data.seq]
    neg_data = neg_data.drop_duplicates(subset=['hash'], keep='last').drop(['hash'],axis = 1)
    print('Mononucleotide negatives were generated!')
    return neg_data

def A2GAlienGeneration(pos_data:pd.DataFrame, tf_name:str, data_dir:str, 
                  num_negatives:int = 2, seed:int=29) -> pd.DataFrame:
    lst_paths = list(filter(lambda x: tf_name not in x, glob.glob(f'{data_dir}**/*.fastq', recursive=True)))
    lst_alien_seq = []

    for path in lst_paths:
        records = [str(i.seq) for i in SeqIO.parse(path, format='fastq')]
        lst_alien_seq.extend(records)
    lst_alien_seq = list(set(lst_alien_seq))
    lst_alien_filtered = []
    lst_replics = []
    lst_hashes = []
    for replic in pos_data.replic.unique():
        selected_aliens = [SeqEntry(seq) for seq in lst_alien_seq]
        sampler = SetGCSampler.make(negatives=selected_aliens, 
                            sample_per_object=num_negatives, 
                            seed=seed)
        selected_positives = [SeqEntry(seq) for seq in pos_data.loc[pos_data.replic == replic,'seq'].values]
        lst_aliens_temp = sampler.sample(positive = selected_positives)
        lst_alien_filtered.extend([seq.sequence for seq in lst_aliens_temp])
        lst_replics.extend([replic] * len(lst_aliens_temp))
        lst_hashes.extend(hash(seq.sequence) for seq in lst_aliens_temp)
        unique_seqs = set(lst_alien_filtered)
        lst_alien_seq = [seq for seq in lst_alien_seq if seq not in unique_seqs]

    neg_data = pd.DataFrame(dict(seq = [i for i in lst_alien_filtered] , 
                                label = np.zeros( len(lst_alien_filtered),dtype= 'int'),
                                cycle = np.zeros( len(lst_alien_filtered),dtype= 'int'),
                                replic = lst_replics, 
                                hash = lst_hashes
                            ))
    
    neg_data = neg_data.drop_duplicates(subset=['hash']).drop(['hash'],axis = 1)
    print('Alien negatives were generated!')
    return neg_data

def A2G_random_gen(pos_data:pd.DataFrame, seed:int = 29, 
                    num_negatives:int = 2,seq_size:int = 40, gc_content:float = 0.41):
    np.random.seed(seed)
    ALPHABET = np.array(list('ATCG'))
    pr = np.array([(1 - gc_content)/2]*2 + [gc_content/2]*2)
    seq_set = set(pos_data.seq.values)
    lst_neg = ['' for i in range(pos_data.shape[0]*num_negatives)]
    size_train = ((pos_data.group.values == 'train').sum()) * num_negatives
    size_val = (pos_data.shape[0] - size_train//num_negatives) * num_negatives

    id_anchor = 0
    while id_anchor != len(lst_neg):
        random_seq = ''.join(np.random.choice(ALPHABET, size = seq_size, p = pr, replace=True))
        if random_seq in seq_set:
            continue
        else:
            lst_neg[id_anchor] = random_seq
            id_anchor += 1

    neg_data = pd.DataFrame(dict(seq = lst_neg, 
                        label = np.zeros( len(lst_neg), dtype='int')), 
                        cycle = np.concatenate([pos_data.cycle.values for i in range(num_negatives)]),
                        replic = np.concatenate([pos_data.replic.values for i in range(num_negatives)]))

    return neg_data

#GHTS and CHS block

def G2A_read(path:str, genome_path:str = GENOME) -> pd.DataFrame:
    if genome_path != GENOME:
        print('Mus musculus!')
    parser = SeqIO.parse(genome_path, format = 'fasta')
    lst_data = []

    for pth in glob.glob(path + '*'):
        bed_like = pd.read_csv(pth, sep='\t')
        lst_data.append(bed_like)
    bed_like = pd.concat(lst_data, axis=0)
    converted = dict(chr = [], seq = [], mid=[], hash=[])
    for id, chr in enumerate(parser):
        name = chr.name
        seqs = bed_like.loc[bed_like.iloc[:,0] == name] #['#CHROM']
        for seq in seqs.itertuples():
            s, e = int(seq[2]), int(seq[3])
            seq = chr[s:e]
            converted['seq'].append(str(seq.seq).upper())
            converted['chr'].append(name)
            converted['hash'].append(hash(str(seq.seq).upper()))
            converted['mid'].append((s+e)//2)
    pos_data = pd.DataFrame(converted).drop_duplicates(subset=['hash']).drop('hash',axis=1,)
    pos_data['label'] = 1
    return pos_data

def G2AAlienGeneration(pos_data:pd.DataFrame, tf_name:str, exp_dir:str, 
                  genome_path:str = GENOME, num_negatives:int = 1,seed:int = 29) -> pd.DataFrame:
    lst_tfs = list(filter(lambda x: x!= tf_name, os.listdir(exp_dir)))
    parser = SeqIO.parse(genome_path,format='fasta')
    lst_data = []
    for tf in lst_tfs:
        path = exp_dir + f'/{tf}'
        for pth in glob.glob(path + '/*'):
            lst_data.append(pd.read_csv(pth,sep='\t'))
        
    whole_data = pd.concat(lst_data, axis=0)
    converted = dict(chr = [], seq = [], hash=[])
    for id, chr in enumerate(parser):
        name = chr.name
        seqs = whole_data.loc[whole_data['#CHROM'] == name]
        for seq in seqs.itertuples():
            s, e = int(seq[2]), int(seq[3])
            seq = chr[s:e]
            converted['seq'].append(str(seq.seq).upper())
            converted['chr'].append(name)
            converted['hash'].append(hash(str(seq.seq).upper()))
    converted = pd.DataFrame(converted).drop_duplicates(subset=['hash']).drop('hash',axis=1)

    available_chroms = pos_data.chr.unique()
    lst_aliens = []
    lst_chr = []
    for chr_id in available_chroms:
        filtered_aliens = converted.loc[converted.chr == chr_id,:]
        filtered_pos = pos_data.loc[pos_data.chr == chr_id,:]
        selected_negatives = [SeqEntry(row[2], metainfo=dict(chr=row[1])) for row in filtered_aliens.itertuples()]
        
        sampler = SetGCSampler.make(negatives=selected_negatives, 
                            sample_per_object=num_negatives, 
                            seed=seed)
        selected_positives = [SeqEntry(row[2], metainfo=dict(chr=row[1])) for row in filtered_pos.itertuples()]
        lst_aliens_temp = sampler.sample(positive = selected_positives)
        lst_aliens.extend([str(entry.sequence) for entry in lst_aliens_temp])
        lst_chr.extend(len(lst_aliens_temp)*[chr_id])

    neg_data = dict(chr = lst_chr,
                   seq = lst_aliens,
                   label =  np.zeros(len(lst_aliens), dtype='int'))
    neg_data = pd.DataFrame(neg_data)
    neg_data['hash'] = [hash(seq) for seq in neg_data.seq.values]
    neg_data = neg_data.drop_duplicates(subset=['hash']).drop('hash',axis=1)
    print('Alien negatives were generated!')
    return neg_data

def G2AShuffleSeq(pos_data:pd.DataFrame, seed:int = 29, num_negatives:int = 1, num_workers:int = 1,
                    shuffle_type:str = 'mono') -> pd.DataFrame:
    '''
    Generation of negative controls with help of shuffle
    '''

    if seed is not None:
        np.random.seed(seed)
    lst_seq = pos_data.seq.values
    lst_neg_seq = []
    lst_chr_unit = pos_data.chr
    lst_chr = []
    shuffle_func = (lambda x: ''.join(np.random.permutation(list(x))) ) if shuffle_type == 'mono' else shuffle_seq_dinucl

    for i in range(num_negatives):
        temp_lst = list(pmap(shuffle_func, seq= lst_seq, threads=num_workers))
        lst_neg_seq.extend(temp_lst)
        lst_chr.extend(lst_chr_unit)
    neg_data = pd.DataFrame({"chr":lst_chr,'seq':lst_neg_seq, 'label':np.zeros(len(lst_neg_seq), dtype = 'int')})
    neg_data['hash'] = [hash(seq) for seq in neg_data.seq]
    neg_data = neg_data.drop_duplicates(subset=['hash'], keep='last').drop(['hash'],axis = 1)
    print('Negatives were generated!')
    return neg_data

def dataset_generation(exp_type:str, tf_name:str, num_workers:int, 
                       path_to_data:str = './train_board/',
                       neg_type:str = 'dinucl', seed:int=29, genome_path:str = './genome/hg38.fa') -> pd.DataFrame:
    '''
    Main function, generates a dataset
    '''
    exp_dir = f'{path_to_data}{exp_type}/'
    path = f'{path_to_data}{exp_type}/{tf_name}/'
    neagative_formula = set(neg_type.split('_'))

    if exp_type == 'HTS':
        pos_data = A2G_read(path)
        all_data_list = [pos_data]

        if 'mono' in neagative_formula:
            all_data_list.append(A2GShuffleSeq(pos_data,num_workers,seed=seed,shuffle_type='mono'))
        if 'dinucl' in neagative_formula:
            all_data_list.append(A2GShuffleSeq(pos_data,num_workers,seed=seed, shuffle_type='dinucl'))
        if 'alien' in neagative_formula:
            all_data_list.append(A2GAlienGeneration(pos_data,  tf_name, exp_dir, seed=seed))
        

    elif exp_type == 'GHTS' or exp_type == 'CHS':
        
        if 'mono' in neagative_formula:
            all_data_list.append(G2AShuffleSeq(pos_data,num_workers,seed=seed,shuffle_type='mono'))
        if 'dinucl' in neagative_formula:
            all_data_list.append(G2AShuffleSeq(pos_data,num_workers,seed=seed,shuffle_type='dinucl'))
        if 'alien' in neagative_formula:
            all_data_list.append(G2AAlienGeneration(pos_data,  tf_name, exp_dir, seed=seed))

    if len(all_data_list) == 0:
        raise KeyError('The type of negatives must be one of [\'alien\',\'shuffle\', \'alien_dinucl\', \'alien_mono\',\'mono\' ]')


    dataset = pd.concat(all_data_list, axis = 0)
    dataset.reset_index(inplace=True, drop=True)

    return dataset

    #     dataset.index = list(range(dataset.shape[0]))
    #     print('Negative controls are being generated!')
    #     if neg_type == 'alien_dinucl':
    #         neg_data_dinucle = A2G_dinucl_gen(pos_data, num_workers, num_negatives = 1)
    #         neg_data_alien = A2GAlienGeneration(pos_data,  tf_name, exp_dir, seed=seed, num_negatives = 1)
    #         neg_data = pd.concat([neg_data_dinucle, neg_data_alien])
    #     elif neg_type == 'alien_mono':            
    #         neg_data_mono = A2G_monoshuffle_gen(pos_data, num_workers, num_negatives = 1)
    #         neg_data_alien = A2GAlienGeneration(pos_data,  tf_name, exp_dir, seed=seed, num_negatives = 1)
    #         neg_data = pd.concat([neg_data_mono, neg_data_alien])
            
    #     elif neg_type == 'alien':
    #         neg_data = A2GAlienGeneration(pos_data, tf_name, exp_dir, seed=seed)
    #     elif neg_type == 'dinucl':
    #         neg_data = A2G_dinucl_gen(pos_data, num_workers)
    #     elif neg_type == 'mono':
    #         neg_data = A2G_monoshuffle_gen(pos_data, num_workers=num_workers, num_negatives=2)
    #     elif neg_type == 'random':
    #         neg_data = A2G_random_gen(pos_data, seed=seed)
    #     else:
    #         raise(KeyError('The type of negatives must be one of [\'alien\',\'shuffle\',\'random\', \'alien_dinucl\', \'alien_mono\',\'mono\' ]'))

    #     dataset = pd.concat([pos_data, neg_data], axis=0)
    #     dataset['hash'] = [hash(seq) for seq in dataset.seq.values]
    #     dataset.sort_values(by = ['cycle'], inplace=True)
    #     dataset = pd.DataFrame(dataset).drop_duplicates(subset=['hash'], keep= 'last').drop('hash',axis=1)
    #     replic_sizes = dataset.replic.value_counts()
        
    #     if len(replic_sizes) > 1 and replic_sizes[0] < replic_sizes[1]:
    #         repl1_idx, repl2_idx = dataset.replic == 0, dataset.replic == 1
    #         dataset.loc[repl1_idx,'replic'], dataset.loc[repl2_idx,'replic'] = (1,0)
    #     replic_sizes = dataset.replic.value_counts()
    #     if len(replic_sizes) > 1:
    #         assert replic_sizes[0] > replic_sizes[1]
    #     print('Negative controls were generated!')

    # elif exp_type in ('GHTS', 'CHS') :
    #     pos_data = G2A_read(path, genome_path = genome_path)
    #     if neg_type == 'alien':
    #         neg_data_alien = G2AAlienGeneration(pos_data, tf_name, exp_dir,seed=seed)
    #     elif neg_type == 'alien_mono':
    #         neg_data_alien = G2AAlienGeneration(pos_data, tf_name, exp_dir)
    #         neg_data_mono = G2AShuffleSeq(pos_data, num_workers=2, shuffle_type='mono')
    #         neg_data = pd.concat([neg_data_alien, neg_data_mono], axis=0)
    #     elif neg_type == 'dinucl':
    #         neg_data = G2AShuffleSeq(pos_data, num_workers=2, shuffle_type='dinucl')
    #     elif neg_type == 'dinucl_shades':
    #         neg_data_dinucle = G2AShuffleSeq(pos_data, num_workers=2, shuffle_type='dinucl')
    #         neg_data_shades = G2A_bad_shades(path, genome_path = genome_path)
    #         neg_data = pd.concat([neg_data_shades,neg_data_shades],axis=0)
        
    #     dataset = pd.concat([pos_data,neg_data], axis = 0)
    #     dataset.index = list(range(dataset.shape[0]))
    # else:
    #     print('Requested experiment type isn\'t supported')
    
    # if dataset is None:
    #     print('You may have had an error in your vatiables. Function returns None.')
    # return dataset