# Look Less Think More: Rethinking Compositional Action Recognition
Pytorch implementation of LLTM

## Get started
### Prerequisite
The neccseearay packages can be installed by the following commonds:
```
conda create -n LLTM python=3.8
conda activate LLTM
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge 
pip install av opencv-contrib-python
pip install tqdm ftfy regex tensorboardX
pip install transformers
```
### Preprocess datasets
- Download [Something-Something Dataset](https://github.com/joaanna/something_else) and [Something-Else Annotations](https://github.com/joaanna/something_else), or you can learn more details for STH-ELSE in [something-else](https://github.com/joaanna/something_else).
- Here we provide two ways to load raw data: videos under ```dataset/video```and frames under ```dataset/STHELSE/frames```, or you can set a suitable path for yourself. We load raw video data when set the parameter ```read_from_video=True```, otherwise load raw frames data.  
- Use ```tools/extract_text.py``` to extract object tags information (```caption_full.json```includes video id and object tags) and label texture information (```label_full.json```includes video id and label texture) from ```train.json``` and ```validation.json```(here raw json file can be downloaded in [something-else](https://github.com/joaanna/something_else)), such as:

    ```
    # object tags
    {
    "42326": "margarine, bread",
    "100904": "pen",
    ...
    }

    # label texture
    {
    "56093": "Uncovering something",
    "211460": "Hitting something with something",
    ...
    }
    ```
- Obtain ```train_videofolder.txt```, ```val_videofolder.txt``` and ```category.txt``` by using following command:

    ```
    bash tools/gen_label_STHELSE.py
    ```
- In this project we load the downloaded pre-trained model locally, which need to be placed under directory ```ops/pretrained_weights```, or you can load these pre-trained weights online.
- More details, such as the data-split settings, could refer to [here](https://drive.google.com/open?id=1XqZC2jIHqrLPugPOVJxCH_YWa275PBrZ).

## Training
Run ```train.sh``` to train the model. 

## Evaluate
For evaluate, you should run ```eval.sh``` to test the performance.

## Contributions
- The core code to implement the Instance-centric Video Mutation is the ```do_mask``` function in ```ops/dataset.py```, and you can decide weather to use mutation alone or mixed mutations by controlling the following parameters:
    ```
    # using mixed mutations
    do_aug_mix=True 
    
    # using mutation alone and a is selected from ['partial_dark', 'blur', 'light', 'color']
    do_aug_mix=False 
    mask_mode=a 
    ```
- To implement the Contrastive Commonsense Association, we first project ```global_tags_feas``` and ```labels_feas``` to a common feature space (same dimension) in ```archs/two_stream_new.py``` and then compute the logits metrixs of them by function ```create_logits``` (in ```trainer.py```). And then we calculate the KL loss to push these two features by setting ```do_KL=True```.
 
## Notes

We give the path of dataset and results in our project, as shown below:

- dataset
  - boundingbox
    + anno.pkl
    + caption_full.json
    + label_full.json
  - STHELSE
    + COM
      - train_videofolder.txt
      - val_videofolder.txt
    + frames
    + category.txt
  - video
- ckpt
  * STHELSE

but you can set the appropriate file path for yourselves.   



 
完整代码 https://pan.baidu.com/s/1jJh9PkLrLRQf8HtEeJKIXA?pwd=gzc9