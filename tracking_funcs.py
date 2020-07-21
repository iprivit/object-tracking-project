import numpy as np
from scipy.spatial import distance
from copy import copy

def get_object_distances(predboxess, verbose=False):
    helmet_track = {}
    helmet_dist = {}
    for i,pr in enumerate(predboxess):
        if(i==0):
            for j,p in enumerate(pr['pred_boxes']):
                helmet_track[j] = p   
        else:
            helmet_dist[i] = {}
            new_coords = []
            for j,p in enumerate(pr['pred_boxes']):
                helmet_dist[i][j] = []
                new_coords.append(p)
                for k in helmet_track:
    #                 try:
                    helmet_dist[i][j].append(distance.euclidean(p,helmet_track[k]))
            mins = []
            for k in helmet_track:
                try:
                    mins.append(np.argmin(helmet_dist[i][k]))
                except:
                    try:
                        helmet_track[k]
                    except:
                        helmet_track[k] = p
            #print('length of minimums:' ,len(mins))
            #print('length of helmet_track',len(helmet_track))
            f_ups = []
            for k in range(len(mins)):
                try:
                    #print('mins[k] is: ', mins[k])
                    #print('np where mins==k is :', np.where(np.array(mins)==k))
                    helmet_track[k] = new_coords[np.where(np.array(mins)==k)[0][0]]
                except:
                    if verbose:
                        print(f'ID {k} DONE Fd UP!!!')
                    f_ups.append(np.where(np.array(mins)==k)[0])
    return helmet_dist

def find_dups(helmet_ids):
    dups = []
    for h in helmet_ids:
        if(len(helmet_ids[h])!=len(set(helmet_ids[h]))):
            dups.append(h)
    return(dups)


def dedupe_helmets(dups, helmet_ids, helmet_dist, verbose=True):
    for d in dups[:]:
        if verbose:
            print('helmet ids: ',helmet_ids[d])

        #find IDs that are duplicates
        seen = {}
        dupes = []

        for x in helmet_ids[d]:
            if x not in seen:
                seen[x] = 1
            else:
                if seen[x] == 1:
                    dupes.append(x)
                seen[x] += 1
        if verbose:
            print('dupes: ',dupes)

        # find indices of IDs that are duplicates
        dup_inds = []
        for du in dupes:
            dup_ind = np.where(helmet_ids[d]==du)
            if verbose:
                print(f'found duplicate indices for {du} at {dup_ind}')
            dup_inds.append(dup_ind)

        if verbose:
            print('dup_inds: ',dup_inds)

        #get distances of duplicate IDs
        for i,dup in enumerate(dup_inds):
            # dup is an array, dup_inds is a list of arrays
            du = dupes[i]
            if verbose:
                print('the duplicate :',dup[0])
            #print(f'helmet dist for helmet ind {dup[0][0]}: ',helmet_dist[d][dup[0][0]])
            min_dist1 = helmet_dist[d][dup[0][0]][du]
            if verbose:
                print(f'minimum distance for helmet ind {dup[0][0]}', min_dist1)
            #print(f'helmet dist for helmet ind {dup[0][1]}: ',helmet_dist[d][dup[0][1]])
            min_dist2 = helmet_dist[d][dup[0][1]][du]
            if verbose:
                print(f'minimum distance for helmet ind {dup[0][1]}', min_dist2)
            if min_dist1<min_dist2:
                orig_val = copy(helmet_dist[d][dup[0][1]][du])
                helmet_dist[d][dup[0][1]][du] = 10000
                new_min = np.argmin(helmet_dist[d][dup[0][1]])
                if(new_min in helmet_ids[d]):
                    new_min = max(helmet_ids[d])+1
                helmet_dist[d][dup[0][1]][du] = orig_val
                if verbose:
                    print('new minimum: ',new_min)
                if verbose:
                    print(f'sanity check on helmet dist you set {helmet_dist[d][dup[0][1]][du]}')
                helmet_ids[d][dup[0][1]] = new_min
            else:
                orig_val = copy(helmet_dist[d][dup[0][0]][du])
                helmet_dist[d][dup[0][0]][du] = 10000
                # compute new minimum after setting real min value to 10k
                new_min = np.argmin(helmet_dist[d][dup[0][0]])
                # if there already exists an index there then just set to new ID
                if(new_min in helmet_ids[d]):
                    new_min = max(helmet_ids[d])+1
                helmet_dist[d][dup[0][1]][du] = orig_val
                if verbose:
                    print('new minimum: ',new_min)
                if verbose:
                    print(f'sanity check on helmet dist you set {helmet_dist[d][dup[0][0]][du]}')
                helmet_ids[d][dup[0][0]] = new_min
    return(helmet_ids)


def remove_dups(dups, helmet_ids, verbose=False):
    while len(dups)>0:
        dups = find_dups(helmet_ids)
        if verbose:
            print(f'dups before {dups}')
        helmet_ids = dedupe_helmets(dups, helmet_ids, verbose=False)
        dups = find_dups(helmet_ids)
        if verbose:
            print(f'dups after {dups}')
    return(dups, helmet_ids)