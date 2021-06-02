import os
import sys
import argparse
from os.path import isfile, join, basename, splitext
import json
import random
import pickle
from tqdm import tqdm

def process_paragraph( section ):
    para = []
    for paragraph in section:
        if not paragraph['cite_spans']:
            para.append( paragraph['text'] ) 
        else:
            cite_spans = [
                (span['start'], span['end'], span['text'], '<|CITE|>') for span in paragraph['cite_spans']
            ]
            new_text = replace_refspans(spans_to_replace=sorted(cite_spans), full_string=paragraph['text'])
            para.append( new_text )
    return para

def replace_refspans(
        spans_to_replace, #: list[tuple[int, int, str, str]],
        full_string: str,
        pre_padding: str = "",
        post_padding: str = "",
        btwn_padding: str = ", "
) -> str:
    """
    For each span within the full string, replace that span with new text
    :param spans_to_replace: list of tuples of form (start_ind, end_ind, span_text, new_substring)
    :param full_string:
    :param pre_padding:
    :param post_padding:
    :param btwn_padding:
    :return:
    """
    # assert all spans are equal to full_text span
    assert all([full_string[start:end] == span for start, end, span, _ in spans_to_replace])

    # assert none of the spans start with the same start ind
    start_inds = [rep[0] for rep in spans_to_replace]
    assert len(set(start_inds)) == len(start_inds)

    # sort by start index
    spans_to_replace.sort(key=lambda x: x[0])

    # form strings for each span group
    for i, entry in enumerate(spans_to_replace):
        start, end, span, new_string = entry

        # skip empties
        if end <= 0:
            continue

        # compute shift amount
        shift_amount = len(new_string) - len(span) + len(pre_padding) + len(post_padding)

        # shift remaining appropriately
        for ind in range(i + 1, len(spans_to_replace)):
            next_start, next_end, next_span, next_string = spans_to_replace[ind]
            # skip empties
            if next_end <= 0:
                continue
            # if overlap between ref span and current ref span, remove from replacement
            if next_start < end:
                next_start = 0
                next_end = 0
                next_string = ""
            # if ref span abuts previous reference span
            elif next_start == end:
                next_start += shift_amount
                next_end += shift_amount
                next_string = btwn_padding + pre_padding + next_string + post_padding
            # if ref span starts after, shift starts and ends
            elif next_start > end:
                next_start += shift_amount
                next_end += shift_amount
                next_string = pre_padding + next_string + post_padding
            # save adjusted span
            spans_to_replace[ind] = (next_start, next_end, next_span, next_string)

    spans_to_replace = [entry for entry in spans_to_replace if entry[1] > 0]
    spans_to_replace.sort(key=lambda x: x[0])

    # apply shifts in series
    for start, end, span, new_string in spans_to_replace:
        assert full_string[start:end] == span
        full_string = full_string[:start] + new_string + full_string[end:]

    return full_string


parser = argparse.ArgumentParser( description = 'split dataset into train, dev, test' )
parser.add_argument('manifest', help='folder with all of the papers in json form' )
parser.add_argument('--mask', '-m', choices = ['mask','citeid'],  default = 'mask', help='Whether to mask the text with a <<CITE>> (mask) or the citation id (citeid).' )
#parser.add_argument('outdir', help='the output directory to put the files into' )

args = parser.parse_args()

manifest = args.manifest
#outdir = args.outdir
mask_type = args.mask

print(args )
alls = []
train = []
dev = []
test = []


alls = {}

#traverse files and extract text
with open( manifest ) as mnfst:
    for i, line in enumerate(tqdm(mnfst)):
        gorc_id = line.split()[0]
        filename = line.split()[1]
        with open( filename ) as f: 
            js_dict = json.load( f )       
            #print(js_dict.keys())
            if "paper_id" in js_dict and "metadata" in js_dict:
                # Mask out text by removing the last references first to preserve order
                # Should not parse if body_text is empty or does not exist
                #print(js_dict['grobid_parse']['grobid_parse'].keys()) 
                if js_dict['grobid_parse'] and 'body_text' in js_dict['grobid_parse'] and 'abstract' in js_dict['grobid_parse']: 

                    abstract = js_dict["grobid_parse"]["abstract"]
                    intro_text = [ item for item in js_dict['grobid_parse']['body_text'] if item["section"] and "intro" in item["section"].lower() ]
                    body_text = js_dict['grobid_parse']['body_text']

                    masked_abstract = process_paragraph( abstract )
                    masked_intro = process_paragraph( intro_text )
                    masked_body = process_paragraph( body_text )
                    #if not masked_abstract:
                    #    print(len(masked_abstract), len(abstract))
                    #    print(abstract) 
                    paper_js = {}
                    paper_js['abstract'] = masked_abstract
                    paper_js['intro_txt'] = masked_intro
                    paper_js['body_txt'] = masked_body
                    #paper_js['intro_masked_txt'] = intro_masked_txt
                    #paper_js['body_masked_txt'] = body_masked_txt
                    
                    paper_id = gorc_id
            
                    alls[paper_id] = paper_js
                    #print( paper_id )  
                   #txt_path =  os.path.join( outdir, base+'.txt' ) 
                   # print( txt_path ) 
                   # with open( txt_path, 'w+' ) as w:
                   #     w.write( masked_txt ) 

with open('dict.pkl', 'wb') as w:
    pickle.dump(alls, w)

with open('dict.json', 'w+') as w:
    json.dump(alls, w)
