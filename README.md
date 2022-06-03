## EfficientFormer<br><sub>Vision Transformers at MobileNet Speed</sub>

[arXiv](https://arxiv.org/abs/2206.01191) 


<p align="center">
  <img src="images/dot.png" width=70%> <br>
  Models are trained on ImageNet-1K and measured by iPhone 12 with CoreMLTools to get latency.
</p>



>[EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)<br>
>[Yanyu Li](https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=XUj8koUAAAAJ&sortby=pubdate)<sup>1,2</sup>, [Genge Yuan](https://scholar.google.com/citations?user=tBIAgtgAAAAJ&hl)<sup>1,2</sup>, [Yang Wen](https://www.linkedin.com/in/yang-wen-76749924/)<sup>1</sup>, [Eric Hu](https://www.linkedin.com/in/erichuju/)<sup>1</sup>, [Georgios Evangelidis](https://sites.google.com/site/georgeevangelidis/)<sup>1</sup>, <br>[Sergey Tulyakov](http://www.stulyakov.com/)<sup>1</sup>, [Yanzhi Wang](https://coe.northeastern.edu/people/wang-yanzhi/)<sup>2</sup>, [Jian Ren](https://alanspike.github.io/)<sup>1</sup>  
><sup>1</sup>Snap Inc., <sup>2</sup>Northeastern University


<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
    Vision Transformers (ViT) have shown rapid progress in computer vision tasks, achieving promising results on various benchmarks. However, due to the massive number of parameters and model design, e.g., attention mechanism, ViT-based models are generally times slower than lightweight convolutional networks. Therefore, the deployment of ViT for real-time applications is particularly challenging, especially on resource-constrained hardware such as mobile devices. Recent efforts try to reduce the computation complexity of ViT through network architecture search or hybrid design with MobileNet block, yet the inference speed is still unsatisfactory. This leads to an important question: can transformers run as fast as MobileNet while obtaining high performance? To answer this, we first revisit the network architecture and operators used in ViT-based models and identify inefficient designs. Then we introduce a dimension-consistent pure transformer (without MobileNet blocks) as design paradigm. Finally, we perform latency-driven slimming to get a series of final models dubbed EfficientFormer. Extensive experiments show the superiority of EfficientFormer in performance and speed on mobile devices. Our fastest model, EfficientFormer-L1, achieves 79.2% top-1 accuracy on ImageNet-1K with only 1.6 ms inference latency on iPhone 12 (compiled with CoreML), which is even a bit faster than MobileNetV2 (1.7 ms, 71.8% top-1), and our largest model, EfficientFormer-L7, obtains 83.3% accuracy with only 7.0 ms latency. Our work proves that properly designed transformers can reach extremely low latency on mobile devices while maintaining high performance. 
</details>


<br>
Code coming soon.



## EfficientFormer Model Zoo

### ImageNet-1K
| Model | Top-1 Acc.| Latency on iPhone 12 (ms) | Pytorch Checkpoint | CoreML |
| :--- | :---: | :---: | :---: |:---: |
| EfficientFormer-L1 | 79.2 (80.2) | 1.6| [L1](https://drive.google.com/file/d/11SbX-3cfqTOc247xKYubrAjBiUmr818y/view?usp=sharing) | [L1-mlmodel](https://drive.google.com/file/d/1m6Znj4I1YLYOveOKtWGskB10hd6e5bx_/view?usp=sharing) |
| EfficientFormer-L3 | 82.4 | 3.0| [L3](https://drive.google.com/file/d/1OyyjKKxDyMj-BcfInp4GlDdwLu3hc30m/view?usp=sharing) | [L3-mlmodel](https://drive.google.com/file/d/12xb0_6pPAy0OWdW39seL9TStIqKyguEj/view?usp=sharing) |
| EfficientFormer-L7 | 83.3  | 7.0| [L7](https://drive.google.com/file/d/1cVw-pctJwgvGafeouynqWWCwgkcoFMM5/view?usp=sharing) | [L7-mlmodel](https://drive.google.com/file/d/1CnhAyfylpvvebT9Yn3qF8vrUFjZjuO3F/view?usp=sharing) |

### Latency Measurement 

The latency reported is based on the open-source [CoreMLTools](https://github.com/apple/coremltools). 

[coreml-performance](https://github.com/vladimir-chernykh/coreml-performance) can simply benchmark the speed of our released mlmodels. Thanks for the nice-implemented latency measurement! 

*Tips*: MacOS+XCode and a mobile device (iPhone 12) are needed to reproduce the reported speed. 


## Citation

If our code or models help your work, please cite our [paper](https://arxiv.org/abs/2206.01191):
```BibTeX
@article{li2022efficientformer,
  title={EfficientFormer: Vision Transformers at MobileNet Speed},
  author={Li, Yanyu and Yuan, Geng and Wen, Yang and Hu, Eric and Evangelidis, Georgios and Tulyakov, Sergey and Wang, Yanzhi and Ren, Jian},
  journal={arXiv preprint arXiv:2206.01191},
  year={2022}
}
```
