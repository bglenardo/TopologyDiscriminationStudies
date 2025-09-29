import uproot as up
import numpy as np
import pandas as pd
import awkward as ak
import os
import sys
import matplotlib.cm as cm
from sklearn.cluster import DBSCAN
import pickle as pkl



class TopologyFunctions:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = self.load_data()

    def load_data(self):
        with up.open(self.file_path) as file:
            tree = file['data']
            print('Converting to np arrays...')
            data = tree.arrays(library='np', entry_stop=-1)
            print('Done.\n')
        return data



    def ClusteringSimple(self, evtidx=0, cluster_scale=1., doubleEvent = False):
        clusters = []
        
        scale = cluster_scale # units: mm
        data = self.data
        
        # Clusters will be dicts with:
        #    - total energy
        #    - mean position
        #    - RMS position
        
        if not doubleEvent:
            x = data['stepX'][evtidx]
            y = data['stepY'][evtidx]
            z = data['stepZ'][evtidx]
            energy = data['stepEnergy_keV'][evtidx]
            time = data['stepTime_ns'][evtidx]
        else:
            x = np.append( data['stepX'][evtidx], data['stepX'][evtidx+1] )
            y = np.append( data['stepY'][evtidx], data['stepY'][evtidx+1] )
            z = np.append( data['stepZ'][evtidx], data['stepZ'][evtidx+1] )
            energy = np.append( data['stepEnergy_keV'][evtidx], data['stepEnergy_keV'][evtidx+1] )
            time = np.append( -data['stepTime_ns'][evtidx], data['stepTime_ns'][evtidx+1] )
        
        # Sorting by time doesn't quite work, since the electron tracks can happen essentially simultaneously (I guess)
        if doubleEvent:
            srtidx = np.argsort(time)
        else:
            srtidx = np.argsort(z)
        x = x[srtidx]
        y = y[srtidx]
        z = z[srtidx]
        energy = energy[srtidx]
        
        this_cluster = {}
        this_energy = [energy[0]]
        this_positions = [np.array([x[0],y[0],z[0]])]
        
        for i in range(1,len(x)):
            
            dist = np.sqrt( (x[i]-x[i-1])**2 + (y[i]-y[i-1])**2 + (z[i]-z[i-1])**2 )
            
            if dist < scale:
                # Add step "i" to the present cluster
                this_energy.append( energy[i] )
                this_positions.append( np.array([x[i], y[i], z[i]]) )
                
            else:
                # End this cluster, compute and store its relevant quantities, and then start a new
                # cluster with step "i"
                this_energy = np.array(this_energy)
                this_positions = np.array(this_positions)
                
                this_cluster['Energy'] = np.sum(this_energy)
                # this_cluster['Position'] = np.array( [ np.sum(this_energy*this_positions[:,0]), \
                #                                       np.sum(this_energy*this_positions[:,1]), \
                #                                       np.sum(this_energy*this_positions[:,2]) ]) / np.sum(this_energy)
                this_cluster['X'] = np.sum(this_energy*this_positions[:,0]) / np.sum(this_energy)
                this_cluster['Y'] = np.sum(this_energy*this_positions[:,1]) / np.sum(this_energy)
                this_cluster['Z'] = np.sum(this_energy*this_positions[:,2]) / np.sum(this_energy)
                this_cluster['Width'] = np.sqrt( np.sum( this_energy* ( (this_positions[:,0] - this_cluster['X'])**2 + \
                                                            (this_positions[:,1] - this_cluster['Y'])**2 + \
                                                            (this_positions[:,2] - this_cluster['Z'])**2 ) ) / np.sum(this_energy) )
                
                clusters.append(this_cluster)
                
                this_cluster = {}
                this_energy = [energy[i]]
                this_positions = [np.array([x[i],y[i],z[i]])]
                
        this_energy = np.array(this_energy)
        this_positions = np.array(this_positions)

        this_cluster['Energy'] = np.sum(this_energy)
        this_cluster['X'] = np.sum(this_energy*this_positions[:,0]) / np.sum(this_energy)
        this_cluster['Y'] = np.sum(this_energy*this_positions[:,1]) / np.sum(this_energy)
        this_cluster['Z'] = np.sum(this_energy*this_positions[:,2]) / np.sum(this_energy)
        # this_cluster['Position'] = np.array( [ np.sum(this_energy*this_positions[:,0]), \
        #                                       np.sum(this_energy*this_positions[:,1]), \
        #                                       np.sum(this_energy*this_positions[:,2]) ]) / np.sum(this_energy)
        this_cluster['Width'] = np.sqrt( np.sum( this_energy* ( (this_positions[:,0] - this_cluster['X'])**2 + \
                                                            (this_positions[:,1] - this_cluster['Y'])**2 + \
                                                            (this_positions[:,2] - this_cluster['Z'])**2 ) ) / np.sum(this_energy) )
        clusters.append(this_cluster)
                
        return pd.DataFrame(clusters)
                
    #####################################################################################################################
    def ClusteringDBSCAN(self, evtidx=0, cluster_scale=1.5, min_samples=1, doubleEvent=False):
        data = self.data
        if not doubleEvent:
            x = data['stepX'][evtidx]
            y = data['stepY'][evtidx]
            z = data['stepZ'][evtidx]
            energy = data['stepEnergy_keV'][evtidx]
        else:
            x = np.append(data['stepX'][evtidx], data['stepX'][evtidx+1])
            y = np.append(data['stepY'][evtidx], data['stepY'][evtidx+1])
            z = np.append(data['stepZ'][evtidx], data['stepZ'][evtidx+1])
            energy = np.append(data['stepEnergy_keV'][evtidx], data['stepEnergy_keV'][evtidx+1])

        points = np.vstack([x, y, z]).T
        labels = DBSCAN(eps=cluster_scale, min_samples=min_samples).fit_predict(points)

        clusters = []
        for lbl in np.unique(labels):
            mask = labels == lbl
            e = energy[mask]
            pos = points[mask]
            cluster = {
                "Energy": np.sum(e),
                "X": np.sum(e * pos[:,0]) / np.sum(e),
                "Y": np.sum(e * pos[:,1]) / np.sum(e),
                "Z": np.sum(e * pos[:,2]) / np.sum(e),
                "Width": np.sqrt(np.sum(e * np.sum((pos - pos.mean(axis=0))**2, axis=1)) / np.sum(e)),
            }
            clusters.append(cluster)

        return pd.DataFrame(clusters)
    
    
    def DataframeToROOT(self, df: pd.DataFrame, filename: str, treename: str = "tree"):
        """
        Convert a pandas DataFrame with array-like columns into a ROOT file with C++ vectors.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame. Columns can be scalars or array-like (lists/numpy arrays).
        filename : str
            Name of the output ROOT file.
        treename : str
            Name of the TTree to be created.
        """

        awkward_arrays = {}
        for col in df.columns:
            coldata = df[col].to_numpy()

            # If entries are array-like, build an awkward array of variable-length lists
            if isinstance(coldata[0], (list, np.ndarray)):
                awkward_arrays[col] = ak.Array(coldata)
            else:
                awkward_arrays[col] = np.array(coldata)

        with up.recreate(filename) as fout:
            fout[treename] = awkward_arrays
            
            

    # def ListOfDFsToROOT(self, df_list: list[pd.DataFrame], filename: str, treename: str = "tree"):
    #     """
    #     Convert a list of per-event DataFrames into a ROOT TTree with vector branches.
        
    #     Parameters
    #     ----------
    #     df_list : list of pd.DataFrame
    #         Each DataFrame contains the data for one event.
    #         All DataFrames must have the same column structure.
    #     filename : str
    #         Output ROOT filename.
    #     treename : str
    #         Name of the TTree to be created.
    #     """

    #     if not df_list:
    #         raise ValueError("Input list of DataFrames is empty")

    #     # Initialize storage: one list per branch
    #     branch_data = {col: [] for col in df_list[0].columns}

    #     # Loop over events
    #     for df in df_list:
    #         for col in df.columns:
    #             values = df[col].to_numpy()

    #             # If this event has multiple rows → store as vector
    #             if len(values) > 1:
    #                 branch_data[col].append(values)
    #             else:
    #                 # Single value → store scalar
    #                 branch_data[col].append(values[0])

    #     # Convert to awkward/numpy arrays
    #     awkward_arrays = {}
    #     for col, vals in branch_data.items():
    #         if isinstance(vals[0], (list, np.ndarray)):
    #             awkward_arrays[col] = ak.Array(vals)
    #         else:
    #             awkward_arrays[col] = np.array(vals)

    #     # Write to ROOT
    #     with up.recreate(filename) as fout:
    #         fout[treename] = awkward_arrays
    def ListOfDFsToROOT(self, df_list: list[pd.DataFrame], filename: str, treename: str = "tree"):
        """
        Convert a list of per-event DataFrames into a ROOT TTree with vector branches.
        Each branch is always written as a std::vector, even if length 1.
        """

        if not df_list:
            raise ValueError("Input list of DataFrames is empty")

        # Initialize storage: one list per branch
        branch_data = {col: [] for col in df_list[0].columns}

        # Loop over events
        for df in df_list:
            for col in df.columns:
                values = df[col].to_numpy()
                # Always store as a list/array (even if single value)
                branch_data[col].append(values.tolist())

        # Convert everything to Awkward Arrays (variable-length lists → std::vector)
        awkward_arrays = {col: ak.Array(vals) for col, vals in branch_data.items()}

        # Write to ROOT
        with up.recreate(filename) as fout:
            fout[treename] = awkward_arrays
                
            
    def ReturnClusteredData( self, cluster_scale = 1., num_evts=-1, doubleEvent=False ):

        data = self.data

        if num_evts > 0 and doubleEvent:
            num_evts = 2*num_evts
        elif num_evts > 0:
            num_evts = num_evts
        else:
            num_evts = len(data['stepX'])

        clusters = []
        
        if doubleEvent:
            for i in range(0,num_evts-2,2):
                if i%1000==0: print('Running signal evt {}...'.format(i))
                clusters.append( self.ClusteringDBSCAN(i, cluster_scale, doubleEvent=doubleEvent) )
        else:
            for i in range(0,num_evts):
                if i%1000==0: print('Running evt {}...'.format(i))
                clusters.append( self.ClusteringDBSCAN(i, cluster_scale, doubleEvent=doubleEvent) )
        
        return clusters
    
    
    
    def ClusterAndWriteToPickle( self, clusters, outfilename, cluster_scale = 2., num_evts=-1, doubleEvent=False ):
        
        self.clusters = self.ReturnClusteredData( cluster_scale = cluster_scale, num_evts=num_evts, doubleEvent=doubleEvent)
        
        with open(outfilename,'wb') as outfile:
            pkl.dump(clusters, outfile)
        

