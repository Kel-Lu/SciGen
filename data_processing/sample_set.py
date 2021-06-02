from tqdm import tqdm, trange
import linecache
import argparse
import random


#   Used to 

def main():
    parser = argparse.ArgumentParser( description="randomly sample a subset of lines from a larger file. Output file will be [inputfulename].sample")
    parser.add_argument('--input_file', help='datafile. Each line should be its own data example')
    parser.add_argument('--nsamples', '-n', type=int, default = 500,  help ='number to sample from. Should be < lines in input file. Defaults to 500.')
    parser.add_argument('--seed', type=int, help = 'seed to set randomizer. optional')
    args = parser.parse_args()
    #set random seed to 5 or something then select lines to use as val
    #random.seed( 5 )
    if args.seed:
        random.seed( args.seed )

    items = range( 91577 )
    random.shuffle( items )
    sampler = items[:args.nsamples]

    lst = []
    for i, line in enumerate(tqdm(sampler)):
        sample = linecache.getline(args.input_file, line)
        lst.append( sample )
    output_file=args.input_file+'.sample'
    with open( output_file, 'w+') as w:
        for item in lst:
            w.write( item  ) 

if __name__ == '__main__':
    main()
            
