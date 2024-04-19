import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing as mp 


MODEL_NAME = "sihaochen/SegmenT5-large"
SEGMENT5_PROMPT = "segment sentence: {}"
SEGMENT5_SEP_TOKEN = "[sep]"
DATASET_PATH = 'processed_part11'
gen_kwargs = {
    "length_penalty": 0,
    "max_new_tokens": 256,
    "min_length": 10,
    "no_repeat_ngram_size": 0,
    "num_beams": 1,
}


def segment_text_batch(texts, model, tokenizer, device):
    inputs = [SEGMENT5_PROMPT.format(text) for text in texts]
    input_ids = tokenizer(inputs, return_tensors="pt", padding="max_length", max_length=512, truncation=True, add_special_tokens=True).input_ids.to(device)
    logits = model.generate(input_ids, **gen_kwargs)
    outputs = tokenizer.batch_decode(logits, skip_special_tokens=True)
    segmented_outputs = [output.split(SEGMENT5_SEP_TOKEN) for output in outputs]

    torch.cuda.empty_cache()
    del logits, input_ids
    return segmented_outputs


def process_file(args, progress_queue):
    try:
        filename, gpu_id = args
        


        file_path = os.path.join(DATASET_PATH, filename)
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)

        if "segmented_article" in data:
            print(f"Skipping already processed file: {filename}")
            # torch.cuda.empty_cache()
            try:
                progress_queue.put("skipped", timeout=2)
            except queue.Full:
                # Handle a full queue scenario
                print(f"Queue is full, waiting to put skipped status for {filename}.")
                time.sleep(1)  # Wait a bit before retrying
                progress_queue.put("skipped", timeout=2)  # Retry
            return
        
        # Process articles
        articles = data.get("article_untok", [])
        if not articles:
            print(f"No articles found in {filename}")
            try:
                progress_queue.put("skipped", timeout=2)
            except queue.Full:
                print(f"Queue is full, waiting to put no articles status for {filename}.")
                time.sleep(1)
                progress_queue.put("skipped", timeout=2)
            return
        
        
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        model.eval()

        print('filename: ', filename)

        if articles:
            article_batches = [articles[i:i + 32] for i in range(0, len(articles), 32)]
            segmented_articles = []
            for batch in article_batches:
                segmented_batch = segment_text_batch(batch, model, tokenizer, device)
                segmented_articles.extend(segmented_batch)

            data["segmented_article"] = segmented_articles

        # Process abstracts
        abstract_untok = data.get("abstract_untok", [])
        if abstract_untok:
            abstract_batches = [abstract_untok[i:i + 32] for i in range(0, len(abstract_untok), 32)]
            segmented_abstract = []
            for batch in abstract_batches:
                segmented_batch = segment_text_batch(batch, model, tokenizer, device)
                segmented_abstract.extend(segmented_batch)
                
            data["segmented_abstract"] = segmented_abstract

        # Process candidates
        candidates_untok = data.get("candidates_untok", [])
        segmented_candidates = []
        for candidate_group in candidates_untok:
            if candidate_group[0]:
                candidate_batches = [candidate_group[0][i:i + 32] for i in range(0, len(candidate_group[0]), 32)]
                segmented_group = []
                for batch in candidate_batches:
                    segmented_batch = segment_text_batch(batch, model, tokenizer, device)
                    segmented_group.extend(segmented_batch)
                segmented_candidates.append(segmented_group)
        if segmented_candidates:
            data["segmented_candidates"] = segmented_candidates


        # Write processed data back to file
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

        # After processing
        try:
            progress_queue.put("done", timeout=2)  # Signal completion with a timeout
        except queue.Full:
            print(f"Queue is full, waiting to put done status for {filename}.")
            time.sleep(1)
            progress_queue.put("done", timeout=2)


    except Exception as e:

        print(f"Error processing file: {e}")
        try:
            progress_queue.put("error", timeout=2)  # Signal an error with a timeout
        except queue.Full:
            print("Queue is full, error status could not be put immediately.")
            time.sleep(1)
            progress_queue.put("error", timeout=2)



def progress_monitor(total_files, progress_queue):
    pbar = tqdm(total=total_files, desc="Processing files")
    for _ in range(total_files):
        msg = progress_queue.get()  # Wait for a message
        pbar.update(1)
    pbar.close()


def main():
    num_gpus = torch.cuda.device_count()
    json_files = [f for f in os.listdir(DATASET_PATH) if f.endswith('.json')]
    
    manager = mp.Manager()
    progress_queue = manager.Queue()  # Use a manager queue for inter-process communication

    file_gpu_pairs = [(file, i % num_gpus) for i, file in enumerate(json_files)]

    # Start the progress monitor
    monitor_proc = mp.Process(target=progress_monitor, args=(len(file_gpu_pairs), progress_queue))
    monitor_proc.start()

    # Processing files
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [executor.submit(process_file, pair, progress_queue) for pair in file_gpu_pairs]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Exception in executor: {e}")

    # Wait for the monitor process to finish
    monitor_proc.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
    

