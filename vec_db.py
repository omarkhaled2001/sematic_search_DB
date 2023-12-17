import struct
import os
from typing import Dict, List, Annotated
import numpy as np
from sklearn.cluster import MiniBatchKMeans


class VecDB:

    def __init__(self, file_path = "PATH_DB_10K", new_db = True) -> None:
        self._file_path = file_path
        if new_db:
            # just open new file to delete the old one
            #with open(self._file_path +"/data_points.npy", "w") as fout:
                # if you need to add any head to the file
                #pass
            pass

    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        #Features Vectors
        embedings_rows=[]
        self.db_size=len(rows)
        for row in rows:
            id, embed = row["id"], row["embed"]
            embedings_rows.append(embed)
        # Create the directory if it doesn't exist
        os.makedirs("./" + self._file_path , exist_ok=True)
        #Saving for loading it in retrive func. 
        np.save("./" + self._file_path + "/data_points", embedings_rows)            
        #Creating 1-Level Indexing File        
        self._build_index(embedings_rows)
        
        
    def retrive(self , query: Annotated[List[float], 5], top_k = 10):
        # Loading centroids do use it in calculating cosine simularity with query vector
        centrioids = np.load("./" + self._file_path + "/centriods.npy",allow_pickle=True)
        scores = []
        for id in range(len(centrioids)):
            # Access the current centroid using its index
            centriod = centrioids[id]
            #calculating cosine simularity
            score = self._cal_score(query, centriod)
            scores.append((score, id))
        # here we assume that if two rows have the same score, return the lowest ID
        index = sorted(scores, reverse=True)[0][1] #closest Centriod index
        ########################TODO : comment this code
        #print ('closest Centriod is : ', index)
        ########################
        # Loading centroids ids points use it in calculating cosine simularity with query vector
        cluster_ids = np.load("./" + self._file_path + "/cluster"+ str(index) + '.npy',allow_pickle=True)
        
        final_scores = []
        
        # Loading chosen points use it in calculating cosine simularity with query vector
        rows=np.load("./"+  self._file_path +"/data_points.npy",allow_pickle=True)
        
        for id in range(len(cluster_ids)):
            # Access the current centroid using its index
            point_id = cluster_ids[id]
            #calculating cosine simularity
            score = self._cal_score(query, rows[point_id])
            final_scores.append((score, point_id))

        final_indexes = sorted(final_scores  , reverse=True)[:top_k] #closest points
        ########################TODO : comment this code
        #print ('closest Points_indexes is : ', [s[1] for s in final_indexes])
        #print ('closest Points score is : ', [s[0] for s in final_indexes])
        ########################            
        return [s[1] for s in final_indexes]        


    # Calculate Cosine Similarity 
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity


     
     

    def _build_index(self,rows: List[Annotated[List[float], 70]]):
        # num_Clusters = Data Size / 2000 #Every Cluster has 2000 record 
        # batch_Size = By sense
        
        batch_Size=100
        n_Clusters=2
        if len(rows)== 10000: #10K
            batch_Size= 2000
            n_Clusters= 10
        elif len(rows)== 100000: #100K
            batch_Size= 50000
            n_Clusters= 50
        elif len(rows)== 1000000: #1M
            batch_Size= 100000
            n_Clusters= 500
        elif len(rows)== 5000000: #5M
            batch_Size= 1000000
            n_Clusters= 2500  
        elif len(rows)== 10000000:#10M
            batch_Size= 1000000
            n_Clusters= 5000
        elif len(rows)== 15000000: #15M
            batch_Size= 1000000
            n_Clusters= 7500 
        elif len(rows)== 20000000: #20M
            batch_Size= 1000000
            n_Clusters= 10000 
            
        batch_begin=0
        ##Begin of learning Phase
        kmeans = MiniBatchKMeans(n_clusters = n_Clusters, batch_size = batch_Size, n_init=1)
        counter = len(rows)//batch_Size -1 # For Loaping over whole Data 
        
        while counter >= 0: #Loaping over whole Data using small batches steps 
            #Update k means estimate on a single mini-batch X.
            kmeans = kmeans.partial_fit(rows[batch_begin:batch_begin+batch_Size])
            batch_begin+=batch_Size
            counter-=1
            
        #Compute cluster centers and predict cluster index for each sample.    
        labels = kmeans.predict(rows)
        #Rearrange Data samples in List of Centiode each one has it's own points ids.
        all_assignments = [[] for _ in range(n_Clusters)]
        for i in range(len(rows)):
                all_assignments[labels[i]].append(i)
        
        #Saving Data samples Centiodes for loading it in retrive func.
        np.save("./" + self._file_path + "/centriods", kmeans.cluster_centers_)
        #Saving Data samples ids in each Centiode for loading it in retrive func.
        for i in range(n_Clusters):
            if not os.path.exists("./" + self._file_path + "/cluster"+str(i)+".npy"):
                np.save("./" + self._file_path + "/cluster"+str(i),all_assignments[i])
            else:
                # Remove the existing file if it exists
                os.remove("./" + self._file_path + "/cluster"+str(i)+".npy")
                # Save your new data directly
                np.save("./" + self._file_path + "/cluster"+str(i),all_assignments[i])
            
        