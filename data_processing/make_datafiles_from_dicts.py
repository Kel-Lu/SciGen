import argparse
import json
from collections import defaultdict
#import spacy
import re
import pickle
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer

parser = argparse.ArgumentParser( description = 'Create a file from a list of examples as: [source context]<SEP>[cited context]<tgt>[citation target]. Context can be abstract, intro, or full (sampled).' )
parser.add_argument('dict', help='dictionary file with all of the papers in json form' )
parser.add_argument('datafile', help='datafile as source_id, source_year, cited_id, cited_year, incites, targeT_sentence')
parser.add_argument('outfile')
parser.add_argument('--source_context', '-s', choices = ['abs', 'intro', 'full'], default ='abs' )
parser.add_argument('--cited_context', '-c', choices = ['abs', 'intro', 'full'], default ='abs' )
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
                
                if not is_valid(source_paper) or not is_valid(cited_paper):
                    continue 
                def flat_list( l ):
                    flat = []
                    for item in l:
                        if type(item) == type([]):
                            for subitem in item:
                                flat.append(subitem)
                        else:
                            flat.append(item)
                    return flat
                #flat_list = lambda l: [item for sublist in l for item in sublist if type(sublist] 
                src_ctx = " ".join(flat_list(papers_dict[source_id][source_context]))
                ctd_ctx = " ".join(flat_list(papers_dict[cited_id][cited_context]))
                #print( src_ctx )
                
                src_ctx = tokenizer.tokenize(src_ctx)[:450]
                ctd_ctx = tokenizer.tokenize(ctd_ctx)[:450]
                target = tokenizer.tokenize(target)[:100]
                 
                src_ctx = tokenizer.convert_tokens_to_string( src_ctx )
                ctd_ctx = tokenizer.convert_tokens_to_string( ctd_ctx )
                target = tokenizer.convert_tokens_to_string( target )

                example = src_ctx + '<|SEP|>' + ctd_ctx + '<|tgt|>' + target + '<|endoftext|>' 
                ids = str(source_id) + '\t' + str(cited_id) + '\t' + target
                lines_to_write.append( example ) 
                ids_to_write.append( ids ) 
            else:
                pass
                #print( source_id, source_id in papers_dict )
                #print( cited_id, cited_id in papers_dict )
                #print( '\n\n')        
                #print( papers_dict[source_id], papers_dict[cited_id], target ) 
with open(out, 'w+') as f_out:
    for item in lines_to_write:
        f_out.write( item.strip() + '\n')
with open( out+'.ids', 'w+') as f_out:
    for item in ids_to_write:
        f_out.write( item.strip() + '\n')
print( ctr ) 

 
