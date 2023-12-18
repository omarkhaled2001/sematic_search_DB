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
            _, embed = row["id"], row["embed"]
            embedings_rows.append(embed)
        # Create the directory if it doesn't exist
        os.makedirs("./" + self._file_path , exist_ok=True)
          
        #Creating 1-Level Indexing File        
        self._build_index(embedings_rows)
        
        
    def retrive(self , query: Annotated[List[float], 5], top_k = 10):
        # Approx. Worst Ram Usage for 20M data Points = 289.001 MB = 20M *(0.03% * (70features * 4Bytes + 4Bytes + 2* (2 score tuple * 4Bytes ))) + 10K * ( 70features * 4Bytes + 2* (2 score tuple * 4Bytes )+ (2 index tuple * 4Bytes ) ) = 303.04 * 10^6 Bytes        
        # Loading centroids do use it in calculating cosine simularity with query vector
        centriods = np.load("./" + self._file_path + "/Cluster_Centriods.npy",allow_pickle=True)
        #print("centriods Dimention is : ", len(centriods)," centriod of ",len(centriods[0])," features.")
        scores = []
        for id in range(len(centriods)):
            # Access the current centroid using its index
            centriod = centriods[id]
            #calculating cosine simularity
            score = self._cal_score(query, centriod)
            scores.append((score, id))
        sorted_scores= sorted(scores, reverse=True)            
        # here we assume that if two rows have the same score, return the lowest ID
        first_index = sorted_scores[0][1] #First closest Centriod index
        ########################TODO : comment this code
        ##print ('First closest Centriod is : ', first_index)
        ########################
        # Loading centroids ids points use it in calculating cosine simularity with query vector
        second_index = sorted_scores[1][1] #Second closest Centriod index
        ########################TODO : comment this code
        #print ('Second closest Centriod is : ', second_index)
        ########################
        # Loading centroids ids points use it in calculating cosine simularity with query vector
        third_index = sorted_scores[2][1] #Third closest Centriod index
        ########################TODO : comment this code
        #print ('Third closest Centriod is : ', third_index)
        ########################
        # Loading centroids ids points use it in calculating cosine simularity with query vector                
        cluster_indexes = np.load("./" + self._file_path + "/Cluster_Indexes.npy",allow_pickle=True)
        
        first_cluster_start_index = cluster_indexes[first_index][0]
        first_cluster_end_index = cluster_indexes[first_index][1]
        #print("first_cluster_start_index is : ",first_cluster_start_index," & first_cluster_end_index ",first_cluster_end_index)
        
        second_cluster_start_index = cluster_indexes[second_index][0]
        second_cluster_end_index = cluster_indexes[second_index][1]
        #print("second_cluster_start_index is : ",second_cluster_start_index," & second_cluster_end_index ",second_cluster_end_index)
        
        third_cluster_start_index = cluster_indexes[third_index][0]
        third_cluster_end_index = cluster_indexes[third_index][1]
        #print("third_cluster_start_index is : ",third_cluster_start_index," & third_cluster_end_index ",third_cluster_end_index)
        
        # Loading chosen points indexes use it in calculating cosine simularity with query vector        
        indexes =[]
        data_points_indexes = np.lib.format.open_memmap("./" + self._file_path + "/data_points_indexes"+".npy", mode='r', dtype='int') 

        first_cluster_data_points_indexes = data_points_indexes[first_cluster_start_index:first_cluster_end_index]
        indexes += first_cluster_data_points_indexes.tolist()
        
        second_cluster_data_points_indexes = data_points_indexes[second_cluster_start_index:second_cluster_end_index]
        indexes += second_cluster_data_points_indexes.tolist()   
        
        third_cluster_data_points_indexes = data_points_indexes[third_cluster_start_index:third_cluster_end_index]
        indexes += third_cluster_data_points_indexes.tolist() 
        
        #print("indexes (First 10) = ", indexes[:10])
        #print("indexes Dimention is : ", len(indexes)," index")
        
        # Loading chosen points use it in calculating cosine simularity with query vector
        
        rows =[]
        data_points = np.lib.format.open_memmap("./" + self._file_path + "/data_points"+".npy", mode='r', dtype='float32' , shape=(len(indexes),70)) 

        first_cluster_data_points = data_points[first_cluster_start_index:first_cluster_end_index]
        rows += first_cluster_data_points.tolist() 
        
        second_cluster_data_points = data_points[second_cluster_start_index:second_cluster_end_index]
        rows += second_cluster_data_points.tolist()   
        
        third_cluster_data_points = data_points[third_cluster_start_index:third_cluster_end_index]
        rows += third_cluster_data_points.tolist()           
        
        #print("sorted rows Data Dimention is : ", len(rows)," point of ",len(rows[0])," features.")
        #print("sorted rows Data (First 10 row)(First 3 feature) = ", [row[:3] for row in rows[:10]])

        final_scores = []
        for i in range(len(rows)):
            #calculating cosine simularity
            score = self._cal_score(query, rows[i])
            final_scores.append((score, indexes[i]))

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
        #Approx. Worst Ram Usage for 20M data Points = 10.657 GB = 20M *(70features * 4Bytes + 70features * 4Bytes + 4Bytes + 4Bytes + 4Bytes) + 10K * ( 70features * 4Bytes + 2indexes * 4Bytes + 4Bytes ) = 1.144292 * 10^10 Bytes
        #print("rows_points Dimention is : ", len(rows)," point of ",len(rows[0])," features.")
        #print("First 10 row_points (First 3 feature): ", [row[:3] for row in rows[:10]])


        # num_Clusters = Data Size / 2000 #Every Cluster has 2000 record 
        # batch_Size = By sense
        
        batch_Size=100
        n_Clusters=10
        if len(rows)== 10000: #10K
            batch_Size= 2000
            n_Clusters= 10
        elif len(rows)== 100000: #100K
            batch_Size= 20000
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
            
        batch_begin = 0
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
        arranged_data_samples = [[] for _ in range(n_Clusters)]
        #To git size of each Cluster
        clusters_size = [0 for _ in range(n_Clusters)]
        for index in range(len(rows)):
                arranged_data_samples[labels[index]].append(index)
                clusters_size[labels[index]] += 1 # increamnt
        #for i in range(n_Clusters):        
            ##print("Cluster [",i,"] : size of points: ", clusters_size[i])
            ##print("Cluster [",i,"] : First 5 points ids: ", arranged_data_samples[i][:5])

        #ReSort Data samples in List of indexes based on Clusters's index arrange from centiod 0 to #n_Clusters each one has it's own points ids.
        sorted_data_sample_indexes =[]  
        sorted_data_sample =[]  
        for cluster in arranged_data_samples:
                for index in cluster:
                    sorted_data_sample.append(rows[index])   #List of Float Embeddings 70 Feature Vectors                     
                    sorted_data_sample_indexes.append(index) #List of Int IDs      
        #print("First 10 sorted points (First 3 feature): ", [row[:3] for row in sorted_data_sample[:10]])
        #print("First 10 sorted points ids: ", sorted_data_sample_indexes[:10])                    
        clusters_start_end_indexes = [[0,0] for _ in range(n_Clusters)] # To read Embeddings 70 Feature Vectors List
        start_index = 0 #num_elemnts
        end_index = 0 #num_elemnts
        #Loop
        for i in range(n_Clusters):
            end_index = start_index  + clusters_size[i] 
            clusters_start_end_indexes[i] = [start_index,end_index] 
            start_index = end_index
        #for i in range(n_Clusters):       
            #print("Cluster [",i,"] : start_index : ",clusters_start_end_indexes[i][0]," & end_index ",clusters_start_end_indexes[i][1])
                
        
        #Saving for loading it in retrive func. 
        np.save("./" + self._file_path + "/data_points", sorted_data_sample)  
        #Saving for loading it in retrive func. 
        np.save("./" + self._file_path + "/data_points_indexes", sorted_data_sample_indexes)         
        #Saving Data samples Centiodes for loading it in retrive func.
        np.save("./" + self._file_path + "/Cluster_Centriods", kmeans.cluster_centers_ )
        #Saving Start & End indexes in each Centiode for loading it in retrive func.
        np.save("./" + self._file_path + "/Cluster_Indexes", clusters_start_end_indexes)
