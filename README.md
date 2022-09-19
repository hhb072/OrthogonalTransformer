# Orthogonal Transformer
This is the project for **Orthogonal Transformer: An Efficient Vision Transformer Backbone with Token Orthogonalization** in NeurIPS 2022.

Code will be released soon.

# Abstract
We present a general vision transformer backbone, called as Orthogonal Transformer, in pursuit of both efficiency and effectiveness. A major challenge for vision transformer is that self-attention, as the key element in capturing long-range dependency, is very computationally expensive for dense prediction tasks (e.g., object detection). Coarse global self-attention and local self-attention are then designed to reduce the cost, but they suffer from either neglecting local correlations or hurting global modeling. We present an orthogonal self-attention mechanism to alleviate these issues. Specifically, self-attention is computed in the orthogonal space that is reversible to the spatial domain but has much lower resolution. The capabilities of learning global dependency and exploring local correlations are maintained because every orthogonal token in self-attention can attend to the entire visual tokens. Remarkably, orthogonality is realized by constructing an endogenously orthogonal matrix that is friendly to neural networks and can be optimized as arbitrary orthogonal matrices. We also introduce Positional MLP to incorporate position information for arbitrary input resolutions as well as enhance the capacity of MLP. Finally, we develop a hierarchical architecture for Orthogonal Transformer. Extensive experiments demonstrate its strong performance on a broad range of vision tasks, including image classification, object detection, instance segmentation and semantic segmentation.

## Citation
	 @inproceedings{huang2022Orthogonal,
	   title={Orthogonal Transformer: An Efficient Vision Transformer Backbone with Token Orthogonalization},
	   author={Huang, Huaibo and Zhou, Xiaoqiang and He, Ran},
	   booktitle={Neural Information Processing Systems (NeurIPS)},	   
	   year={2022},
	  }
