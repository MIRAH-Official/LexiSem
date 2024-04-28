from simcse import SimCSE
import json
import torch
import os
import numpy as np
# from torch import Tensor, device

from tqdm import tqdm

import torch
torch.cuda.empty_cache()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")


def read_json_files(folder_path):
    
    
    # Iterate through each JSON file in the directory
    for filename in tqdm(os.listdir(folder_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            
            # Load the JSON data from the file
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            #segmented_article & candidates
            abstract_untok = data['segmented_abstract'] 
            candidates_untok = data['segmented_candidates']

            # abstract_embedding = [abstract_untok][0] 
           
            
            all_candidates_scores = []
            for x, abstract in enumerate(abstract_untok):
                all_scores_abstract = []
                for one_abstract in abstract:
                    one_sent_scores_abstract = []
                    # print("one_abstract", one_abstract)
                    for i, candidate in enumerate(candidates_untok):
                        # print(i, 'candidate', candidate)
                        one_cand_scores = []
                        for n, one_cand in enumerate(candidate): #0 cand
                            one_sent_scores = []
                            # print(n, "one_cand", one_cand )
                            for sentence in one_cand:
                                # print('sentence', sentence)
                                
                                # candidate_texts = candidate[0]
        
                                # Compute similarity between article and candidate texts
                                similarity = model.similarity(one_abstract, sentence, device=device)
                                
                                one_sent_scores.append(similarity)
                        
                        # Compute the max of each row
                            max_of_each_row = np.max(one_sent_scores, axis=0)
                            one_cand_scores.append(max_of_each_row)
                            
                        # Compute the average of these max values
                        average_max_similarity = np.mean(one_cand_scores)
                        one_sent_scores_abstract.append(average_max_similarity)
                    all_candidates_scores.append(one_sent_scores_abstract)
            print('one_sent_scores_abstract', all_candidates_scores)
                

            result_scores = []
            index_offset = 0  # To keep track of our position in all_candidates_scores

            for sublist in abstract_untok:
                sublist_length = len(sublist)
                
                if sublist_length == 1:
                    # If there's only one sentence, take the next sublist from all_candidates_scores as is
                    result_scores.append(all_candidates_scores[index_offset])
                    index_offset += 1
                else:
                    # If there are multiple sentences, calculate the max scores for each corresponding position
                    # across the next 'sublist_length' sublists in all_candidates_scores
                    max_scores = []
                    for i in range(len(all_candidates_scores[index_offset])):
                        # Extract the same index across the next 'sublist_length' sublists and find the max
                        current_scores = [all_candidates_scores[j + index_offset][i] for j in range(sublist_length) if i < len(all_candidates_scores[j + index_offset])]
                        if current_scores:  # Ensure there's something to take the max of
                            max_scores.append(max(current_scores))
                    result_scores.append(max_scores)
                    index_offset += sublist_length  # Move the offset by the length of the current sublist

            print(result_scores)



            # Find the length of the longest sublist to know how many indices we have
            max_length = max(len(sublist) for sublist in result_scores)

            # Calculate the mean for each position across all sublists
            means = []
            for i in range(max_length):
                # Collect all items at position i from each sublist, if available
                items_at_index = [sublist[i] for sublist in result_scores if i < len(sublist)]
                
                # Calculate the mean of these items if there are any
                if items_at_index:
                    mean_value = sum(items_at_index) / len(items_at_index)
                    means.append(mean_value)

            print(means)

            
            for i ,sublist in enumerate(candidates_untok):
                # for x, sublist_scores in enumerate(means):
                    # sublist_length = len(sublist)
                    candidates_untok[i].append(means[i])    

    
    
         
            
            # Optionally, you can save the updated data back to a JSON file
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
                




folder_path = '/home/aloraini131985/all/sub-sentence-encoder/dataset-test1'
read_json_files(folder_path)



