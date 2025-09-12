# CocoER: Aligning Multi-Level Feature by Competition and Coordination for Emotion Recognition
CocoER is a multi-level image feature refinement method for emotion recognition to mitigate the impact caused by conflicting results from multi-level recognition. First, CocoER leverages cross-level  attention to improve visual feature consistency between hierarchically cropped head, body and context windows.  Then, vocabulary informed alignment is incorporated into the recognition framework to produce pseudo label and guide hierarchical visual feature refinement. To effectively fuse multi-level feature, CocoER elaborates on a competition process of eliminating irrelevant  image level predictions and a coordination process to enhance the  feature across all levels. Extensive experiments are executed on two popular datasets, and CocoER achieves state-of-the-art performance with multi-level interpretation results.


<img src="/assets/Asset_1.png" alt="SVG" width="70%" height="70%">


------
### Installation

1. Create conda environment and activate:

   ```bash
   conda create -n cocoer python=3.7
   conda activate cocoer
   ```

2. Install specific versions of PyTorch and related libraries and install dependencies from `requirements.txt`:

   ```bash
   pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
   
   pip install -r requirements.txt
   ```
------
### Model Weight Download

Please download the model weights and place them in the following directories:

* D-Link: 

- GWT model: `./checkpoints/`
- GWT model (VI weights): `./checkpoints/VI_weights/`

For third-party model:

- insightface D-link: https://drive.google.com/file/d/15vdafPSWhep4PJAAYSdSPPgDoCgp2dxO/view?usp=sharing 
- Unzip the file and move the folder `insightface` into  `./thirdparty/`

### Inference

Run the inference script with:

```bash
python inference.py --input ./test_imgs/o1.png
```

* Files starting with "o" in the  `./test_imgs/` folder are images that are not inside the EMOTIC test folder.

### Output Dictionary Explanation

- `output_dict['pred']`: Final emotion  prediction result of the model.
- `output_dict['VI']`: Emotion prediction result of the V-I module.
- `output_dict['head']`: Head-level emotion prediction result.
- `output_dict['body']`: Body-level emotion prediction result.
- `output_dict['ctx']`: Context-level emotion prediction result.
- `output_dict['emo_process']`: Exclusion sequence.

Files with bounding boxes is saved in the `./outputs/` directory. For example:

<img src="/outputs/o1.png" alt="S" width="50%" height="50%">

Have Fun ~~



### Limitation

Due to the constraints of the training dataset, the model may perform poorly on out-of-domain images. 



------
### Citation
```
@InProceedings{Shen_2025_CVPR,
    author    = {Shen, Xuli and Cai, Hua and Shen, Weilin and Xu, Qing and Yu, Dingding and Ge, Weifeng and Xue, Xiangyang},
    title     = {CocoER: Aligning Multi-Level Feature by Competition and Coordination for Emotion Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {29591-29600}
}
```

