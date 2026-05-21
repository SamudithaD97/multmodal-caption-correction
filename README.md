## How to Run

### 1. Create environment

```bash
conda create -n dl_env python=3.10
conda activate dl_env
```

### 2. Install dependencies

```pip install torch torchvision transformers pycocotools pillow```


### 3. Build COCO dataset

```bash
mkdir -p coco/images coco/annotations
cd coco

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip -d images/
unzip annotations_trainval2017.zip

cd ..
```

### 4. Build dataset (generate pairs) - 
#### skip this as I am pushing the jsonl in the commit but keep the image location as mentioned in the manifests. i.e coco-data/images/train2017/000000237912.jpg. Your downloaded coco data should be in the root folder of the project

```bash
python scripts/build_coco_pairs.py \
  --ann-file coco/annotations/captions_train2017.json \ 
  --images-root coco/images/train2017 \ 
  --out-dir data/coco_correction \
  --num-source-images 10000
  ```

### 5. Test the dataloader
`python src/test_dataset_with_batching.py`

### Next step
We have completed creating false caption and true caption for each image and have batched the data in train.jsonl by loading the images and normalizing it which is now ready to be fed to ResNet model and the captions are converted to the tokens using the base tokenizer. Now we build the model and define the training loop with loss. 


### 6. To actually run the training/testing or inference:
- Assuming you're in the IU Quartz cluster - run `sbatch ./scripts/run_sbatch_job.sh`. This will schedule the job and give the job id.
- Change the last line by uncommenting either the test/train/infer/saliency/gradcam line to run the corresponding task.
- You can inspect the logs being printed by streaming the file in the path `./logs/unet-{job_id}.out`. To stream `tail -f ./logs/job_{job_id}.out`. `

