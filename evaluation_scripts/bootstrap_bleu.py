import sacrebleu
import sys
import argparse
import random 
from scipy import stats
import numpy as np
from tqdm import tqdm 
from rouge import Rouge 


parser = argparse.ArgumentParser( description = 'bootstrap bleu' )
parser.add_argument( 'refs' )
parser.add_argument( 'sys1' )
parser.add_argument( 'sys2' )
parser.add_argument( '--size', type=int, default=100 )
parser.add_argument('--samples', type=int, default=1000)

args = parser.parse_args()

def load_sentences( filename ):
    sents = []
    with open( filename ) as f:
        for line in f:
            sents.append( line )
    return sents

refs = load_sentences( args.refs )
sys1 = load_sentences( args.sys1 )
sys2 = load_sentences( args.sys2 )

bleus1 = []
bleus2 = []
rouge1sA = []
rouge1sB = []
rouge2sA = []
rouge2sB = []
rougeLsA = [] 
rougeLsB = []

rouge  = Rouge()

for i in tqdm(range( args.samples )):
    sys1_sampl, sys2_sampl, ref_sampl = zip( *random.sample( list( zip( sys1, sys2,  refs ) ), args.size ) )
    bleuA = sacrebleu.corpus_bleu(sys1_sampl, [ref_sampl])
    bleuB = sacrebleu.corpus_bleu(sys2_sampl, [ref_sampl])
    rouge1 = rouge.get_scores( sys1_sampl, ref_sampl )[0] 
    rouge2 = rouge.get_scores( sys2_sampl, ref_sampl )[0]

    
    r1_A = rouge1["rouge-1"]["f"]
    r1_B = rouge2["rouge-1"]["f"]
    
    r2_A = rouge1["rouge-2"]["f"]
    r2_B = rouge2["rouge-2"]["f"]

    rL_A = rouge1["rouge-l"]["f"]
    rL_B = rouge2["rouge-l"]["f"]

    rouge1sA.append( r1_A ) 
    rouge1sB.append( r1_B )
    
    rouge2sA.append( r2_A )
    rouge2sB.append( r2_B ) 
    
    rougeLsA.append( rL_A )
    rougeLsB.append( rL_B ) 
    

    bleus1.append( bleuA.score )
    bleus2.append( bleuB.score )

get_stats = lambda a: (np.mean(a), np.std(a))        
 
#print( bleu.score )
print('========= BLEU ===========')
print( "blues A ::", get_stats( bleus1) )  
print( "bleus B ::", get_stats(bleus2) )
print( stats.ttest_rel( bleus1, bleus2 ) )

blue1_tot = sacrebleu.corpus_bleu(sys1, [refs])
blue2_tot = sacrebleu.corpus_bleu(sys2, [refs])

print( blue1_tot.score, blue2_tot.score ) 


print( '========= ROUGE =============' )
print( stats.ttest_ind( rouge1sA, rouge1sB ) )
print( stats.ttest_ind( rouge2sA, rouge2sB) )
print( stats.ttest_ind( rougeLsA, rougeLsB) )

print( "rouge1 A ::", get_stats(rouge1sA))
print( "rouge1 B ::", get_stats(rouge1sB))

print( "rouge2 A ::", get_stats(rouge2sA))
print( "rouge2 B ::", get_stats(rouge2sB))

print( "rougeL A ::", get_stats(rougeLsA))
print( "rougeL B ::", get_stats(rougeLsB))

rouge1 = rouge.get_scores( sys1, refs )[0]
rouge2 = rouge.get_scores( sys2, refs )[0]
print ('==========')
print( rouge1)
print( rouge2)


'''
print( len( refs), len( sys ) ) 

bleu = sacrebleu.corpus_bleu(sys, [refs])
print(bleu.score) 
'''

