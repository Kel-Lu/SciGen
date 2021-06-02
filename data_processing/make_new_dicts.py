import argparse
import json
from collections import defaultdict
import spacy
import re
import pickle
import os
from tqdm import tqdm
from random import shuffle

parser = argparse.ArgumentParser( description = 'split dataset into train, dev, test' )
parser.add_argument('dict', help='dictionary file with all of the papers in json form' )
parser.add_argument('manifest', help = 'manifest file')
#parser.add_argument('datafile', help='datafile as source_id, source_year, cited_id, cited_year, incites, targeT_sentence')
parser.add_argument('outfile')
#parser.add_argument('--source_context', '-s', choices = ['abs', 'intro', 'full'], default ='abs' )
#parser.add_argument('--cited_context', '-c', choices = ['abs', 'intro', 'full'], default ='abs' )
args = parser.parse_args()

papers_dict = pickle.load( open(args.dict, 'rb'))
datafile = args.manifest
out = args.outfile

ctr = 0
new_dict = {}
lines_to_write = []
with open( datafile) as f_in:
    for i, line in enumerate( tqdm( f_in) ):
        with open( line.strip() ) as f:
            js_dict = json.load( f ) 
            #print(js_obj.keys() ) 
            pid = js_dict['paper_id']
            if pid in papers_dict and js_dict['grobid_parse'] and 'body_text' in js_dict['grobid_parse'] and 'abstract' in js_dict['grobid_parse']:
                new_dict[pid] = papers_dict[pid].copy()
                body_text = js_dict['grobid_parse']['body_text']
                if body_text:
                #print( type(body_text[0]))
                    sents = []
                    for para in body_text:
                        for sent in para:
                            sents.append(sent)
                    sampled_lst = sents.copy()
                    shuffle(sampled_lst) 
                    new_dict[pid]['body_txt'] = sampled_lst
                    #print( papers_dict[pid].keys() ) 
                    if not papers_dict[pid]['intro_txt']:
                        new_intro = sents
                        #print( new_intro ) 
                        new_dict[pid]['intro_txt'] = new_intro
                    if not papers_dict[pid]['abstract'] or 'metadata' in js_dict:
                        if 'title' in js_dict['metadata'] and 'abstract' in js_dict['metadata']:
                            #print(js_dict['metadata']['title'])
                            #print(  js_dict['metadata']['abstract'] )
                            if js_dict['metadata']['title'] and  js_dict['metadata']['abstract']:
                                if isinstance(  js_dict['metadata']['abstract'], list ):
                                    new_dict[pid]['abstract'] = [js_dict['metadata']['title']] + js_dict['metadata']['abstract']
                                elif isinstance(  js_dict['metadata']['abstract'], str ):
                                    new_dict[pid]['abstract'] = [js_dict['metadata']['title']] + [js_dict['metadata']['abstract']]
                                else:
                                    print( type(js_dict['metadata']['abstract']))
                            else:
                                new_dict[pid]['abstract'] = js_dict['grobid_parse']['abstract']
                                #try:
                                #    print( type(js_dict['grobid_parse']['abstract']), type( js_dict['metadata']['abstract']) ) 
                                #except:
                                #    print( 'title' in js_dict['metadata'], 'abstract' in js_dict['metadata'] ) 
                        else:
                            new_dict[pid]['abstract'] = js_dict['grobid_parse']['absract']
            else:
                #print( type(pid), pid )
                ctr += 1
print( ctr ) 
with open(out, 'wb') as w:
    pickle.dump(new_dict, w)

 
