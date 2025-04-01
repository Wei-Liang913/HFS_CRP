import numpy as np
import pandas as pd
import random
from sklearn.cluster import KMeans
import time

start_time = time.time()
random.seed(0)
np.random.seed(0)

#Parameters
m = 4
n = 3
p = 0 #number of DMs. It would be assigned in the function readPreferences()
theta = 0.5
cd_over = 0.82
r_max = 5

##### READ EXPERTS PREFERENCES FROM FILE #####
def readPreferences():
    global p
    prefs = []
    with open('output_100.txt', 'r') as file:
        for line in file:
            rows = line.split(";")
            # Expert matrix
            e = []
            for row in rows:
                # HFEs
                row = row.strip()
                hfes = row.split(":")
                matrix = [list(map(float, hfe.split(","))) for hfe in hfes]
                e.append(matrix)
            prefs.append(e)
            p = p+1
    file.close()
    return prefs

### TRANSFORM THE FORMAT OF THE PREFERENCES FOR USING K-MEANS
def transformPreferencesClustering(prefs):
    prefsClus = []
    for i in range(len(prefs)):
        prefExp = []
        for j in range(len(prefs[i])):
            for k in range(len(prefs[i][j])):
                prefExp.append(prefs[i][j][k][0])
        prefsClus.append(prefExp)
    
    return prefsClus

##### SIMILARITY BETWEEN EXPERTS #####
def computeSimilarityExperts(e1, e2):
    acum = 0
    for i in range(m):
        for j in range(n):
            acum = acum + abs(e1[i][j][0] - e2[i][j][0]) #there is just one element, (change in the future)
    return 1 - ((1/(m*n)) * acum)


##### COMPUTE SUBGROUPS OPINIONS #####

def perform_kmeans_clustering(prefsClus, m):
    prefsClus = np.array(prefsClus) 
    kmeans = KMeans(n_clusters=m, random_state=42) 
    labels = kmeans.fit_predict(prefsClus) 
    centroids = kmeans.cluster_centers_  
    # print(labels)
    return labels, centroids

def buildClustersKMeans(labels, prefs):
    clusters = [[] for _ in range(max(labels) + 1)]  
    for i, label in enumerate(labels):
        clusters[label].append(prefs[i])
    return clusters

# collect HFEs of all the experts for each cluster in a single matrix
def mergeHFEsByCluster(clusters):
    subgs = []
    for c in clusters:
        #initialize subgroup matrix
        subg = []
        for _ in range(m):
            v_row = [0]*n
            subg.append(v_row)
        #set the HFEs of all experts of a cluster for each alternative/criterion
        for i in range(m):
            for j in range(n):
                n_hfs = []
                for prefs in range(len(c)):
                    for hfe in c[prefs][i][j]:
                        n_hfs.append(hfe)
                subg[i][j] = n_hfs
        subgs.append(subg)
    
    return subgs

# compute statistical information of subgroups
def computeStatisticalInfoByCluster(subgs):
    stats_s = []
    for subg in subgs:
        #initialize matrix of expected/variance values (two values for each cell)
        n_subg = []
        for _ in range(m):
            v_row = [0]*n
            n_subg.append(v_row)
            
        for i in range(m):
            for j in range(n):
                hfes = subg[i][j]
                n_subg[i][j] = [np.round(np.mean(hfes),4), np.round(np.var(hfes),4)]
        print(pd.DataFrame(n_subg))
        stats_s.append(n_subg)

    return stats_s

# compute statistical information of collective opinion
def computeStatisticalInfoCollective(stats_s):
    # initialize collective matrix of expected/variance values (two values for each cell)
    stats_c = []
    for _ in range(m):
        v_row = [0]*n
        stats_c.append(v_row)

    for i in range(m):
        for j in range(n):
            exp_g = 0
            var_g = 0
            for stats in stats_s:
                exp_g += stats[i][j][0]
                var_g += np.sqrt(stats[i][j][1])
            stats_c[i][j] = [np.round(exp_g/len(stats_s),4), np.round(pow(var_g/len(stats_s), 2),4)]
    print(pd.DataFrame(stats_c))
    return stats_c

def computeStatisticalInfobyAlt(stats_c):
    stats_alt = []
    for _ in range(m):
        v_row = [[0,0]]
        stats_alt.append(v_row)

    for i in range(m):
        exp_g = 0
        var_g = 0
        for j in range(n):
            exp_g += stats_c[i][j][0]
            var_g += np.sqrt(stats_c[i][j][1])
        stats_alt[i][0] = [np.round(exp_g / n, 4), np.round(pow(var_g / n, 2),4)]
    return stats_alt

def comparison(stats_alt):
    scores_value = []
    for i in range(m):
        exp = stats_alt[i][0][0]
        var = stats_alt[i][0][1]
        score = np.round(exp / np.sqrt(var), 4)
        scores_value.append((score, i, stats_alt[i][0]))


    sorted_scores = sorted(scores_value, key=lambda x: x[0], reverse=True)
    ranks = {index: rank + 1 for rank, (_, index, _) in enumerate(sorted_scores)}

    for score_, index, data in scores_value:
        rank = ranks[index]
        print(data, score_, rank)

# ##### CONSENSUS PROCESS #####
prefs = readPreferences()
prefsClus = transformPreferencesClustering(prefs)
# print(prefsClus)
result = perform_kmeans_clustering(prefsClus, m)

round = 0
while(True):
    print()
    print()
    print("==========================")
    print("ROUND ", round)
    print("==========================")
    print()

    
    print("CLUSTERS")
    print("==========================")
    
    clusters = buildClustersKMeans(result[0], prefs)
    cluster_indices = []
    for cluster in clusters:
        indices = []
        for expert in cluster:
            indices.append(prefs.index(expert) + 1) 
        cluster_indices.append(indices)
    subgroups = mergeHFEsByCluster(clusters)


    print()
    print("STATISTICAL INFORMATION SUBGROUPS")
    print("==========================")
    stats_subg = computeStatisticalInfoByCluster(subgroups)
    
    print()
    print("STATISTICAL INFORMATION COLLECTIVE")
    print("==========================") 
    stats_col = computeStatisticalInfoCollective(stats_subg)

    # compute consensus degree for each subgroup (alternative/criterion)
    CD_tij = [] # consensus matrices for each subgroup
    CD_ti = [] # consensus degree alternatives for each subgroup
    CD_t = [] # consensus degree for each subgroup
    
    print()
    print("CONSENSUS LEVEL BY SUBGROUP FOR CRITERIA")
    print("==========================")

    for stats in stats_subg:
        CD_ij = np.zeros((m,n), dtype=float)
        for i in range(m):
            for j in range(n):
                CD_ij[i][j] = 1-np.sqrt(pow(stats[i][j][0]-stats_col[i][j][0],2) + 
                                         pow(stats[i][j][1]-stats_col[i][j][1],2))
        CD_tij.append(CD_ij)
        
        print(pd.DataFrame(CD_ij))
        
        # compute consensus degree for each subgroup (alternative)
        CD_i = np.sum(CD_ij, axis=1)/n
        CD_ti.append(CD_i)
        # compute consensus degree for each subgroup
        CD_t.append(np.sum(CD_i)/m)
    
    print()
    print("CONSENSUS LEVEL BY SUBGROUP")
    print("==========================")
    print(CD_t)
    
    # compute overall consensus
    CD = np.round(np.mean(CD_t),2) # here equal weights are considered
    print()
    print("OVERALL CONSENSUS")
    print("==========================")
    print(CD)
    print()

    if CD < cd_over and round < r_max:
        # feedback process
        # identify subgroup
        G_index = CD_t.index(min(CD_t)) 
        G_min = stats_subg[G_index] # subrgroup with the minimum CD
        
        print("Cluster to modify: ", (G_index+1))

        # identify alternative
        CD_i = CD_ti[G_index]
        alt_mod = []
        alt_index = 0
        for cd_i in CD_i:
            if cd_i < cd_over:
                alt_mod.append(alt_index)
            alt_index = alt_index + 1

        print("Alternatives to modify: ", np.array(alt_mod)+1)  
        
        # identify criteria
        CD_ij = CD_tij[G_index] # consensus matrix for the subgroup
        cri = []
        for alt in alt_mod:
            v_cri = CD_ij[alt][:]
            for j in range(len(v_cri)):
                if v_cri[j] < cd_over:
                    cri.append(j)
        
        cri_not_repeated = list(set(cri))
        print("Criteria to modify: ", np.array(cri_not_repeated)+1)
        
        # advices
        G_pref = clusters[G_index]
        for pref in G_pref:
            for alt in alt_mod:
                for c in cri_not_repeated:
                    prev = pref[alt][c]
                    mod = np.array(prev) * (1-theta) + (theta * stats_col[alt][c][0])
                    pref[alt][c] = [mod[0]]
        round += 1
        #print(prefs)

        prefsClus = transformPreferencesClustering(prefs)
        # print(prefsClus)
        result = perform_kmeans_clustering(prefsClus, m)
    else:
        break #consensus is reached, we finish the CRP


print()
print("RESULTS CRP")
print("==========================")
print("Number of rounds: ", round)
print("Consensus achieved: ", CD)
print()
print()

print("==========================")
print("AGGREGATED PREFERENCES BY ALTERNATIVE")
stats_alt = computeStatisticalInfobyAlt(stats_col)
print()

print("==========================")
print("RANKING ORDER")
comparison(stats_alt)
print()
