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

* D-Link: https://pan.baidu.com/s/1ziaZirfyxff-JETBXgKd7Q?pwd=t523

- GWT model: `./checkpoints/`
- GWT model (VI weights): `./checkpoints/VI_weights/`

### Inference

Run the inference script with:

```bash
python inference.py --input ./test_imgs/o2.png
```

* Files starting with "o" in the  `./test_imgs/` folder are images that are not inside the test folder.

### Output Dictionary Explanation

- `output_dict['pred']`: Final emotion  prediction result of the model.
- `output_dict['VI']`: Emotion prediction result of the V-I module.
- `output_dict['head']`: Head-level emotion prediction result.
- `output_dict['body']`: Body-level emotion prediction result.
- `output_dict['ctx']`: Context-level emotion prediction result.
- `output_dict['emo_process']`: Exclusion sequence.

Files with bounding boxes is saved in the `./outputs/` directory.






------
### Showcase:
(1) The V-I column shows pseudo label in vocabulary-informed module. We display the elimination of inconsistent recognition results in orange, which are not shown in either the V-I column or the  prediction column.


<img src="/assets/Asset_2.png" alt="S" width="90%" height="90%">

(2) We provide failure cases of GPT-4o to illustrate that VLM also suffers from conflicting results of multi-level predictions for emotion recognition, by comparing w/ and w/o multi-level inputs. We use orange color to highlight the wrong predictions that both appear in GPT-4o results, but are not contained in ground-truth label.  It indicates that if there are inconsistent predictions at a certain image level, it can also confuse the final recognition results for VLM. Therefore, VLM frameworks also need to eliminate multi-level pollution. 


<img src="/assets/Asset_3.png" alt="S" width="100%" height="100%">




------
### Limitation

Due to the constraints of the training dataset, the model may perform poorly on out-of-domain images. 



Have Fun~


------
### Citation
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

