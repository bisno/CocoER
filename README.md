# CocoER: Aligning Multi-Level Feature by Competition and Coordination for Emotion Recognition
CocoER is a multi-level image feature refinement method for emotion recognition to mitigate the impact caused by conflicting results from multi-level recognition. First, CocoER leverages cross-level  attention to improve visual feature consistency between hierarchically cropped head, body and context windows.  Then, vocabulary informed alignment is incorporated into the recognition framework to produce pseudo label and guide hierarchical visual feature refinement. To effectively fuse multi-level feature, CocoER elaborates on a competition process of eliminating irrelevant  image level predictions and a coordination process to enhance the  feature across all levels. Extensive experiments are executed on two popular datasets, and CocoER achieves state-of-the-art performance with multi-level interpretation results.


<img src="/assets/Asset_1.png" alt="SVG" width="70%" height="70%">

------
### Showcase:
(1) The V-I column shows pseudo label in vocabulary-informed module. We display the elimination of inconsistent recognition results in orange, which are not shown in either the V-I column or the  prediction column.


<img src="/assets/Asset_2.png" alt="S" width="90%" height="90%">

(2) We provide failure cases of GPT-4o to illustrate that VLM also suffers from conflicting results of multi-level predictions for emotion recognition, by comparing w/ and w/o multi-level inputs. We use orange color to highlight the wrong predictions that both appear in GPT-4o results, but are not contained in ground-truth label.  It indicates that if there are inconsistent predictions at a certain image level, it can also confuse the final recognition results for VLM. Therefore, VLM frameworks also need to eliminate multi-level pollution. 


<img src="/assets/Asset_3.png" alt="S" width="100%" height="100%">




------
### To-Do:

- [ ] release inference code.
- [ ] release training code.
- [ ] release checkpoints.


------



### Coming Soon ~
