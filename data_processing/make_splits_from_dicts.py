import argparse
import json
from collections import defaultdict
#import spacy
import re
import pickle
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer
import random

parser = argparse.ArgumentParser( description = 'split dataset into train, dev, test' )
parser.add_argument('dict', help='dictionary file with all of the papers in json form' )
parser.add_argument('datafile', help='datafile as source_id, source_year, cited_id, cited_year, incites, targeT_sentence')
parser.add_argument('outfile')
parser.add_argument('--source_context', '-s', choices = ['abs', 'intro', 'full'], default ='abs' )
parser.add_argument('--cited_context', '-c', choices = ['abs', 'intro', 'full'], default ='abs' )
parser.add_argument('--seed', type = int )
args = parser.parse_args()

papers_dict = pickle.load( open(args.dict, 'rb'))
datafile = args.datafile
out = args.outfile
source_context = args.source_context
cited_context = args.cited_context

to_dict ={
    'abs':'abstract',
    'intro':'intro_txt',
    'full':'body_txt',
}

def is_valid( doc ):
    for key in to_dict:
        val = to_dict[key]
        if not doc[val]:
            return False
    return True    


source_context = to_dict[source_context] 
cited_context = to_dict[cited_context]

#print( papers_dict.keys() ) 
special_tokens = {"additional_special_tokens": ["<|tgt|>", "<|CITE|>"]}

tokenizer = GPT2Tokenizer.from_pretrained('gpt2',
                                            #do_lower_case=args.do_lower_case,
                                            cache_dir= None,
                                            pad_token='<|PAD|>'   ,
                                            sep_token='<|SEP|>',
                                            )#additional_special_tokens=special_tokens,)
tokenizer.add_special_tokens( special_tokens )
ctr = 0
lines_to_write = []
ids_to_write = []


source_documents = []
cited_documents = []
lines = []
with open( datafile) as f_in:
    for i, line in enumerate( tqdm( f_in) ):
        items = ( line.split('\t'))
        if len(items) == 6:
            source_id = (items[0])
            cited_id = (items[2])
            target = items[-1]             
            if source_id in papers_dict and cited_id in papers_dict:
                #print(papers_dict[source_id].keys())
                #example = papers_dict[source_id]['abstract'] + '\t' + 
                #print(papers_dict[source_id][source_context])
                
                source_paper = papers_dict[source_id]
                cited_paper = papers_dict[cited_id]
                
                if is_valid(source_paper) and is_valid(cited_paper):
                    source_documents.append( source_id )
                    cited_documents.append( cited_id )                    
                    lines.append( line ) 

      
source_set = set( source_documents ) 
cited_set = set( cited_documents )


print( type( source_set) )
if args.seed:
    random.seed( args.seed )
for_eval = random.sample( list(source_set),1000 )

test_set = random.sample( for_eval, 500 )
dev_set = set(for_eval) - set( test_set) 
train_set = set(source_set) - set(for_eval)

with open('sampled_test.ids', 'w') as w:
    for item in test_set:
        w.write( str(item) + '\n' )

with open('sampled_dev.ids', 'w') as w:
    for item in dev_set:
        w.write( str(item) + '\n' )

test_lines = []
dev_lines = []

for line in lines:
    items = ( line.split('\t'))
    if len(items) == 6:
        source_id = (items[0])
        cited_id = (items[2])
        target = items[-1]
        if source_id in papers_dict and cited_id in papers_dict:
            if (source_id not in for_eval) and (cited_id not in for_eval):
                def flat_list( l ):
                    flat = []
                    for item in l:
                        if type(item) == type([]):
                            for subitem in item:
                                flat.append(subitem)
                        else:
                            flat.append(item)
                        return flat
                lines_to_write.append( line ) 
            elif source_id in test_set: 
                test_lines.append( line )
            elif source_id in dev_set:
                dev_lines.append( line ) 
            else:
                pass
                #print( source_id, source_id in papers_dict )
                #print( cited_id, cited_id in papers_dict )
                #print( '\n\n')        
                #print( papers_dict[source_id], papers_dict[cited_id], target ) 
with open(out, 'w+') as f_out:
    for item in lines_to_write:
        f_out.write( item.strip() + '\n')

with open( 'sampled_dev.txt', 'w+') as f_out:
    for item in dev_lines:
        f_out.write( item.strip() + '\n' )

with open( 'sampled_test.txt', 'w+') as f_out:
    for item in test_lines:
        f_out.write( item.strip() + '\n' )

with open( out+'.ids', 'w+') as f_out:
    for item in ids_to_write:
        f_out.write( item.strip() + '\n')
print( ctr ) 



 
