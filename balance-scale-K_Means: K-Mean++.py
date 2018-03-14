import csv
import random
from math import sqrt
from math import fsum
import itertools
import math

gold_standard={}
inp = {}
k = 3

with open('Balance-Scale.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    j = 1
    for row in readCSV:
        dict = {}
        iris_class = row[-1]
        for i in range(0, len(row)-1):
            dict[i] = float(row[i]);
        inp[j] = dict
        j += 1
        gold_standard.setdefault(iris_class, [])
        gold_standard[iris_class].append(j)
    '''print(inp)
    print("****************************************************------**********************************************************")
    print(inp.values())'''

def initialize_kmeansplusplus(inputs, k):
    centroids = random.sample(inputs, 1)
    while(len(centroids) < k):
        D2 = []
        for x in inputs:
            distance = 100000000000
            for y in centroids:
                curr_dist = calculate_euclidean_distance(x, y)
                if(curr_dist < distance):
                    distance = curr_dist
            D2.append(distance**2)
        probs = []
        s = fsum(D2)
        for d in D2:
            probs.append(d/s)
        cum_sum = []
        cum_sum.append(probs[0])
        for i in range(1, len(probs)):
            cum_sum.append(cum_sum[i-1]+probs[i])
        rand = random.random()
        index = -1
        for i in range(0, len(cum_sum)):
            if(cum_sum[i] >= rand):
                index = i
                break
        centroids.append(inputs[index])
    return centroids


def Hamming_distance(current_clusters, gold_standard):
        #gold_standard = self.gold_standard
        current_clustes_pairs = []

        for (current_cluster_key, current_cluster_value) in current_clusters.items():
            tmp = list(itertools.combinations(current_cluster_value, 2))
            current_clustes_pairs.extend(tmp)
        tp_fp = len(current_clustes_pairs)

        gold_standard_pairs = []
        for (gold_standard_key, gold_standard_value) in gold_standard.items():
            tmp = list(itertools.combinations(gold_standard_value, 2))
            gold_standard_pairs.extend(tmp)
        tp_fn = len(gold_standard_pairs)


        total = math.factorial(len(inp))/ (math.factorial(len(inp)-2) * math.factorial(2))
        #print(total)
        tp = 0.0
        for ccp in current_clustes_pairs:
            if ccp not in gold_standard_pairs:
                tp += 1
        #print(tp)
        hamming_distance = (tp)/total
        print("Hamming Distance for the cluster is ",hamming_distance)
        #return hamming_distance


def has_converged(new_centroids, old_centroids):
    return (set([tuple(a) for a in new_centroids]) == set([tuple(a) for a in old_centroids]))

def calculate_euclidean_distance(x, centroid):
    sqdist = 0.0
    for i, v in x.items():
        sqdist += (v-centroid[i]) ** 2
    return sqrt(sqdist)

def initialize_centroids(inputs, k):
    centroids = random.sample(inputs, k)
    return centroids

def distribute_points(inputs, centroids):
    clusters = {}
    for x in inputs:
        min = 10000000000
        best_centroid = -1
        for i in range(0, len(centroids)):
            dist = calculate_euclidean_distance(x, centroids[i])
            if(dist < min):
                min = dist
                best_centroid = i
        if best_centroid in clusters:
            clusters[best_centroid].append(x)
        else:
            clusters[best_centroid] = [x]
    return clusters

def reevaluate_centers(centroids, clusters):
    new_centroids = []
    for k in clusters.keys():
        num_rows = len(clusters[k][0])
        new_centroids.append(mean(clusters[k], num_rows))
    return new_centroids

def mean(cluster, l):
    centroid = [0.] * l
    n = 0
    for x in cluster:
        for i, v in x.items():
            centroid[i] += v
        n += 1
    for i in range(0, l):
        centroid[i] /= n
    return centroid

def form_clusters(inputs, centroids):
    clusters = {}
    for y, x in inputs.items():
        min = 10000000000
        best_centroid = -1
        for i in range(0, len(centroids)):
            dist = calculate_euclidean_distance(x, centroids[i])
            if(dist < min):
                min = dist
                best_centroid = i
        if best_centroid in clusters:
            clusters[best_centroid].append(y)
        else:
            clusters[best_centroid] = [y]
    print("")
    Hamming_distance(clusters,gold_standard)
    print("")
    return clusters

def calculate_k_means_cost(inputs, centroids):
    cost = 0
    for y, x in inputs.items():
        min = 10000000000
        best_centroid = -1
        for i in range(0, len(centroids)):
            dist = calculate_euclidean_distance(x, centroids[i])
            if(dist < min):
                min = dist
                best_centroid = i
        cost += min ** 2
    return cost


#print(" ------------------K- means Plus Plus----------------------")
def find_clusters(inpu, k):
    inputs = list(inp.values())
    old_centroids = []
    clusters = {}
    centroids = initialize_kmeansplusplus(inputs, k)
    print("**********************************************************************************************************************")
    print("--------------------------------------Clustering Algorithm---------------\ K - Means ++ /-----------------------------")
    print("**********************************************************************************************************************")
    print("--------********************************-- K - Means ++ Algorithm  Centroids Cluster & HAMMING DISTANCE --***********************-------- : ")
    print("Initial Centroids : ", centroids)
    while not has_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = distribute_points(inputs, centroids)
        centroids = reevaluate_centers(old_centroids, clusters)
        print("Centroid after reevaluating clusters :")
        print(centroids)
        print("****************************************************************************************************************")
    print(" Final Centroids :",centroids)
    print("---***************************************-----------------------------*********************************************")
    clusters = form_clusters(inpu, centroids)
    for x, y in clusters.items():
        print("Cluster ",x+1," is ",y)
        print("")
#print(initialize_centroids(inputs, k))
#print(find_clusters(inp, k))

#print(" --------------------Lloyd's Algorithm---------------------")
def find_clusters_lloyd(inpu, k):
    inputs = list(inp.values())
    old_centroids = []
    clusters = {}
    centroids = initialize_centroids(inputs, k)
    #print("-----------------------------------------------------------")
    #print("Initial Centroids :", centroids)
    while not has_converged(centroids, old_centroids):
        old_centroids = centroids
        clusters = distribute_points(inputs, centroids)
        centroids = reevaluate_centers(old_centroids, clusters)
        #print("Centroid is shifting, New Centroids :")
        #print(centroids)
        #print("-------------------------------------------")
    #print("Final Centroids :",centroids)
    #print("-----------------------------------------------------------")
    #return form_clusters(inp, centroids)
    return centroids
#print(initialize_centroids(inputs, k))
#print(find_clusters_lloyd(inp, k))

def run_lloyds(inpu, k):
    min_k_means_cost = 100000000000000000000
    best_centroids = []
    for i in range(0, 100):
        centroids = find_clusters_lloyd(inp, k)
        print("")
        print(i," Iteration centroid ",centroids)
        k_means_cost = calculate_k_means_cost(inp, centroids)
        print("")
        print(i, " k means cost", k_means_cost)
        if(k_means_cost < min_k_means_cost):
            min_k_means_cost = k_means_cost
            best_centroids = centroids
    print("***************************************************************************************************************************************")
    print("--------------------------------------Clustering Algorithm---------------\ Lloyd - Algorithm ++ /--------------------------------------")
    print("***************************************************************************************************************************************")
    print("--------************************--Lloyd's Algorithm Final Centroid  & Min K-Means COST & HAMMING DISTANCE--*****************-------- : ")
    print("")
    print("Output of Multiple Runs for the Lloyd Algorithm")
    print("")
    print("min_k_means_cost ",min_k_means_cost)
    print("")
    print("Final Centroids", best_centroids)
    print("")
    clusters = form_clusters(inpu, best_centroids)
    for x, y in clusters.items():
        print("Cluster ",x+1," is ",y)
        print("")

print(run_lloyds(inp, k))
print(find_clusters(inp, k))
