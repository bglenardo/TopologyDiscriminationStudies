# Python script which accepts a data file as an argument, performs clustering on the events, and saves the clustered data to a new file which is also given as an argument.
import argparse 
import pickle as pkl
import TopologyFunctions as TF

argument_parser = argparse.ArgumentParser(description='Clustering script which takes in a data file and outputs a clustered data file.')
argument_parser.add_argument('input_file', type=str, help='Input data file (ROOT format).')
argument_parser.add_argument('output_file', type=str, help='Output clustered data file (pkl format).')
argument_parser.add_argument('--num_evts', type=int, default=-1., help='Number of events to cluster. Default is -1, which means all events.')
argument_parser.add_argument('--cluster_scale', type=float, default=1., help='Clustering scale factor. Default is 2.')
argument_parser.add_argument('--doubleEvent', action='store_true', help='Whether to treat events as double events (for signal). Default is False.')
args = argument_parser.parse_args() 


input_file = args.input_file
output_file = args.output_file
num_evts = args.num_evts
cluster_scale = args.cluster_scale
doubleEvent = args.doubleEvent




output_object = TF.TopologyFunctions(input_file)
clustered_data = output_object.ReturnClusteredData(cluster_scale=cluster_scale, num_evts=2000, doubleEvent=doubleEvent)
with open(output_file, 'wb') as fout:
    pkl.dump(clustered_data, fout)

    print('Clustered data saved to {}'.format(output_file))
    
output_object.ListOfDFsToROOT(clustered_data, output_file.replace('.pkl', '.root'), 'clusters')

    
    