import sys
import math
import os
import heapq
import itertools
import csv

class Clustering:
    def __init__(self, ipt_data, ipt_k):
        self.input_file_name = ipt_data
        self.k = ipt_k
        self.dataset = None
        self.dataset_size = 0
        self.dimension = 0
        self.heap = []
        self.clusters = []
        self.gold_standard = {}

    def initialize_heap(self):
        self.heap = []

    #Function used to initialize cluster from file and check parameters
    def initialize(self):
        if not os.path.isfile(self.input_file_name):
            self.quit("Input file does not exist")

        self.dataset, self.clusters, self.gold_standard = self.load_data(self.input_file_name)
        self.dataset_size = len(self.dataset)

        if self.dataset_size == 0:
            self.quit("Input file doesn't contain any data")

        if self.k == 0:
            self.quit("k = 0, 0/No cluster will be generated")

        if self.k > self.dataset_size:
            self.quit("k value is larger than the number of existing clusters")

        self.dimension = len(self.dataset[0]["data"])

        if self.dimension == 0:
            self.quit("dimension for dataset cannot be zero")


    #Function used to calculate the Euclidean Disctance
    def euclidean_distance(self, data_point_one, data_point_two):
        size = len(data_point_one)
        result = 0.0
        for i in range(size):
            f1 = float(data_point_one[i])   # feature for data one
            f2 = float(data_point_two[i])   # feature for data two
            tmp = f1 - f2
            result += pow(tmp, 2)
        result = math.sqrt(result)
        return result

    #Function used to calculate Pairwise Distance in Cluster
    def compute_pairwise_distance(self, dataset):
        result = []
        dataset_size = len(dataset)
        for i in range(dataset_size-1):    # ignore last i
            for j in range(i+1, dataset_size):     # ignore duplication
                dist = self.euclidean_distance(dataset[i]["data"], dataset[j]["data"])
                result.append( (dist, [dist, [[i], [j]]]) )

        return result

    def build_priority_queue(self, distance_list):
        heapq.heapify(distance_list)
        self.heap = distance_list
        return self.heap


    def compute_centroid(self, dataset, data_points_index):
        size = len(data_points_index)
        dim = self.dimension
        centroid = [0.0]*dim
        for idx in data_points_index:
            dim_data = dataset[idx]["data"]
            for i in range(dim):
                centroid[i] += float(dim_data[i])
        for i in range(dim):
            centroid[i] /= size
        return centroid

    #Function used to calculate Single Linkage Clustering
    def Single_Linkage_clustering(self):
        dataset = self.dataset
        current_clusters = self.clusters

        old_clusters = []
        heap = hc.compute_pairwise_distance(dataset)
        heap = hc.build_priority_queue(heap)
        dummy_clusters = {}
        while len(current_clusters) > self.k:
            dist, min_item = heapq.heappop(heap)
            pair_data = min_item[1]
            if not self.valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = {}
            new_cluster_elements = sum(pair_data, [])
            new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)

            new_cluster_elements.sort()
            new_cluster.setdefault("centroid", new_cluster_cendroid)
            new_cluster.setdefault("elements", new_cluster_elements)

            for pair_item in pair_data:
                old_clusters.append(pair_item)
                dummy_clusters[str(pair_item)] = current_clusters[str(pair_item)]
                del current_clusters[str(pair_item)]

            self.single_add_heap_entry(heap, new_cluster, current_clusters, dummy_clusters)
            current_clusters[str(new_cluster_elements)] = new_cluster

        print(current_clusters)
        return current_clusters

    #Function used to calculate Complete Linkage Clustering
    def Complete_Linkage_clustering(self):
        dataset = self.dataset
        current_clusters = self.clusters

        old_clusters = []
        heap = hc.compute_pairwise_distance(dataset)
        heap = hc.build_priority_queue(heap)
        dummy_clusters = {}
        while len(current_clusters) > self.k:
            dist, min_item = heapq.heappop(heap)
            pair_data = min_item[1]
            if not self.valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = {}
            new_cluster_elements = sum(pair_data, [])
            new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)

            new_cluster_elements.sort()
            new_cluster.setdefault("centroid", new_cluster_cendroid)
            new_cluster.setdefault("elements", new_cluster_elements)

            for pair_item in pair_data:
                old_clusters.append(pair_item)
                dummy_clusters[str(pair_item)] = current_clusters[str(pair_item)]
                del current_clusters[str(pair_item)]

            self.complete_add_heap_entry(heap, new_cluster, current_clusters, dummy_clusters)
            current_clusters[str(new_cluster_elements)] = new_cluster
        return current_clusters

    #Function used to calculate Average Linkage Clustering
    def Average_Linkage_clustering(self):
        dataset = self.dataset
        current_clusters = self.clusters
        old_clusters = []
        heap = hc.compute_pairwise_distance(dataset)
        heap = hc.build_priority_queue(heap)
        dummy_clusters = {}
        while len(current_clusters) > self.k:
            dist, min_item = heapq.heappop(heap)
            pair_data = min_item[1]

            if not self.valid_heap_node(min_item, old_clusters):
                continue

            new_cluster = {}
            new_cluster_elements = sum(pair_data, [])
            new_cluster_cendroid = self.compute_centroid(dataset, new_cluster_elements)

            new_cluster_elements.sort()
            new_cluster.setdefault("centroid", new_cluster_cendroid)
            new_cluster.setdefault("elements", new_cluster_elements)

            for pair_item in pair_data:
                old_clusters.append(pair_item)
                dummy_clusters[str(pair_item)] = current_clusters[str(pair_item)]
                del current_clusters[str(pair_item)]

            self.average_add_heap_entry(heap, new_cluster, current_clusters, dummy_clusters)
            current_clusters[str(new_cluster_elements)] = new_cluster

        return current_clusters

    def valid_heap_node(self, heap_node, old_clusters):
        pair_dist = heap_node[0]
        pair_data = heap_node[1]
        for old_cluster in old_clusters:
            if old_cluster in pair_data:
                return False
        return True

    # Function used to calculate Heap Entry for Single Linkage
    def single_add_heap_entry(self, heap, new_cluster, current_clusters, dummy_clusters):
        for ex_cluster in current_clusters.values():
            dist = 10000000
            new_heap_entry = []
            for elem in new_cluster["elements"]:
                li = [elem]
                temp = self.euclidean_distance(ex_cluster["centroid"], dummy_clusters[str(li)]["centroid"])
                if(temp < dist):
                    dist = temp
            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
            heapq.heappush(heap, (dist, new_heap_entry))

    # Function used to calculate Heap Entry for Complete Linkage
    def complete_add_heap_entry(self, heap, new_cluster, current_clusters, dummy_clusters):

        for ex_cluster in current_clusters.values():
            dist = 0
            new_heap_entry = []

            for elem in new_cluster["elements"]:
                li = [elem]
                temp = self.euclidean_distance(ex_cluster["centroid"], dummy_clusters[str(li)]["centroid"])
                if(temp > dist):
                    dist = temp
            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
            heapq.heappush(heap, (dist, new_heap_entry))

    #Function used to calculate Heap Entry for Average Linkage
    def average_add_heap_entry(self, heap, new_cluster, current_clusters, dummy_clusters):

        for ex_cluster in current_clusters.values():
            dist = 0
            new_heap_entry = []
            count =0

            for elem in new_cluster["elements"]:
                li = [elem]
                temp = self.euclidean_distance(ex_cluster["centroid"], dummy_clusters[str(li)]["centroid"])
                dist +=temp
                count+=1
            dist = dist/count
            new_heap_entry.append(dist)
            new_heap_entry.append([new_cluster["elements"], ex_cluster["elements"]])
            heapq.heappush(heap, (dist, new_heap_entry))

    #Function used to calculate Hamming Distance
    def Hamming_distance(self, current_clusters):
        gold_standard = self.gold_standard
        current_clustes_pairs = []

        for (current_cluster_key, current_cluster_value) in current_clusters.items():
            tmp = list(itertools.combinations(current_cluster_value["elements"], 2))
            current_clustes_pairs.extend(tmp)
        tp_fp = len(current_clustes_pairs)

        gold_standard_pairs = []
        for (gold_standard_key, gold_standard_value) in gold_standard.items():
            tmp = list(itertools.combinations(gold_standard_value, 2))
            gold_standard_pairs.extend(tmp)
        tp_fn = len(gold_standard_pairs)


        total = math.factorial(len(self.dataset))/ (math.factorial(len(self.dataset)-2) * math.factorial(2))

        tp = 0.0
        for ccp in current_clustes_pairs:
            if ccp not in gold_standard_pairs:
                tp += 1

        hamming_distance = (tp)/total

        return hamming_distance

    #Function used to load data
    def load_data(self, input_file_name):
        with open(input_file_name) as csvfile:
            readCSV = csv.reader(csvfile, delimiter=',')
            dataset = []
            clusters = {}
            gold_standard = {}
            id = 0
            for row in readCSV:
                iris_class = row[-1]

                data = {}
                data.setdefault("id", id)
                d = []
                for i in range(len(row)-1):
                    d.append(row[i])

                data.setdefault("data", d)
                data.setdefault("class", row[-1])
                dataset.append(data)

                clusters_key = str([id])
                clusters.setdefault(clusters_key, {})
                clusters[clusters_key].setdefault("centroid", row[:-1])
                clusters[clusters_key].setdefault("elements", [id])

                gold_standard.setdefault(iris_class, [])
                gold_standard[iris_class].append(id)

                id += 1

        return dataset, clusters, gold_standard

    def display(self, current_clusters, hamming_distance):
        print("Hamming Distance :" ,hamming_distance)

        clusters = current_clusters.values()
        for cluster in clusters:
            cluster["elements"].sort()
            print(cluster["elements"])

if __name__ == '__main__':


    hc = Clustering("iris_data.csv", 3)
    hc.initialize()
    print("***********************************************************************************************************************")
    print("----------------------------------Clustering Algorithm:--------Single Linkage Clustering-------------------------------")
    print("***********************************************************************************************************************")
    new_clusters = hc.Single_Linkage_clustering()
    hamming_distance = hc.Hamming_distance(new_clusters)
    hc.display(new_clusters, hamming_distance)
    print("***********************************************************************************************************************")
    hc.initialize()
    print("-----------------------------------Clustering Algorithm:-------Complete Linkage Clustering-----------------------------")
    print("***********************************************************************************************************************")
    current_clusters1 = hc.Complete_Linkage_clustering()
    hamming_distance1 = hc.Hamming_distance(current_clusters1)
    hc.display(current_clusters1, hamming_distance1)
    print("***********************************************************************************************************************")
    hc.initialize()
    print("-----------------------------------Clustering Algorithm:--------Average Linkage Clustering-----------------------------")
    print("***********************************************************************************************************************")
    current_clusters2 = hc.Average_Linkage_clustering()
    hamming_distance2 = hc.Hamming_distance(current_clusters2)
    hc.display(current_clusters2, hamming_distance2)
