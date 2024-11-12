# EEG-DCNet

**Updates**: 

* The regularization parameters of [DCNet](https://doi.org/10.1109/TII.2022.3197419) have been modified, resulting in an enhancement in the model's performance and fortifying it against overfitting.
* The current [*main_TrainTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainTest.py) file, following the training and evaluation method outlined in [Paper 1](https://doi.org/10.1109/TII.2022.3197419) and [paper 2](https://ieeexplore.ieee.org/document/10142002), has been identified as not aligning with industry best practices. In response, we strongly recommend adopting the methodology implemented in the refined [*main_TrainValTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainValTest.py) file. This updated version splits the data into train/valid/test sets, following the guidelines detailed in this [post](https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html#) ([Option 2](https://braindecode.org/stable/auto_examples/model_building/plot_how_train_test_and_tune.html#option-2-train-val-test-split:~:text=Number%20of%20samples.%27%3E-,Option%202%3A%20Train%2DVal%2DTest%20Split,-When%20evaluating%20different)). 
##
In addition to the proposed [DCNet](https://doi.org/10.1109/TII.2022.3197419) model, the [*models.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/models.py) file includes the implementation of other related methods, which can be compared with [DNet](https://doi.org/10.1109/TII.2022.3197419), including:
* **EEGNet**, [[paper](https://arxiv.org/abs/1611.08024), [original code](https://github.com/vlawhern/arl-eegmodels)]
* **EEG-TCNet**, [[paper](https://arxiv.org/abs/2006.00622), [original code](https://github.com/iis-eth-zurich/eeg-tcnet)]
* **TCNet_Fusion**, [[paper](https://doi.org/10.1016/j.bspc.2021.102826)]
* **MBEEG_SENet**, [[paper](https://doi.org/10.3390/diagnostics12040995)]
* **EEGNeX**, [[paper](https://arxiv.org/abs/2207.12369), [original code](https://github.com/chenxiachan/EEGNeX)]
* **DeepConvNet**, [[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), [original code](https://github.com/braindecode/braindecode)]
* **ShallowConvNet**, [[paper](https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.23730), [original code](https://github.com/braindecode/braindecode)]
* **EEGATCNet**, [[paper](https://doi.org/10.1109/TII.2022.3197419), [original code](https://github.com/Altaheri/EEG-ATCNet)]

The following table shows the performance of [DCNet]and other reproduced models based on the methodology defined in the [*main_TrainValTest.py*] file:

<table>
    <tr>
        <td rowspan="2">Model</td>
        <td rowspan="2">#params</td>
        <td rowspan="2">#FLOP</td>
        <td colspan="2">BCI Competition IV-2a dataset (<a href="https://www.bbci.de/competition/iv/#dataset2a">BCI 4-2a</a>) </td>
        <td colspan="2">BCI Competition IV-2b dataset (<a href="https://www.bbci.de/competition/iv/#dataset2b">BCI 4-2b</a>) </td>
        <td colspan="2">High Gamma Dataset (<a href="https://github.com/robintibor/high-gamma-dataset">HGD</a>)<sup>*</sup></td>
    </tr>
    <tr>
        <td>accuracy</td>
        <td>kappa</td>
        <td>accuracy</td>
        <td>kappa</td>
        <td>accuracy</td>
        <td>kappa</td>
    </tr>
    <tr>
        <td>DCNet</td>
        <td>28,640</td>
        <td>49</td>
        <td>87.94</td>
        <td>83.92</td>
        <td>92.43</td>
        <td>84.86</td>
        <td>94.55</td>
        <td>92.73</td>
    </tr>
    <tr>
        <td>ATCNet</td>
        <td>113,732</td>
        <td>60.5</td>
        <td>84.80</td>
        <td>79.73</td>
        <td>89.41</td>
        <td>78.80</td>
        <td>92.05</td>
        <td>89.40</td>
    </tr>
    <tr>
        <td>EEGNeX</td>
        <td>63,626</td>
        <td>444</td>
        <td>84.01</td>
        <td>78.68</td>
        <td>86.81</td>
        <td>73.65</td>
        <td>87.58</td>
        <td>83.44</td>
    </tr>
    <tr>
        <td>EEGTCNet</td>
        <td>4,096</td>
        <td>14.2</td>
        <td>77.97</td>
        <td>70.63</td>
        <td>83.69</td>
        <td>67.31</td>
        <td>87.80</td>
        <td>83.73</td>
    </tr>
    <tr>
        <td>MBEEG_SENet</td>
        <td>10,170</td>
        <td>71.5</td>
        <td>79.98</td>
        <td>73.30</td>
        <td>86.53</td>
        <td>73.02</td>
        <td>90.13</td>
        <td>86.84</td>
    </tr>
    <tr>
        <td>ShallowConvNet</td>
        <td>47,310</td>
        <td>127</td>
        <td>80.52</td>
        <td>74.02</td>
        <td>86.02</td>
        <td>72.38</td>
        <td>87.00</td>
        <td>82.67</td>
    </tr>
    <tr>
        <td>EEGNet</td>
        <td>2,548</td>
        <td>26.7</td>
        <td>77.68</td>
        <td>70.24</td>
        <td>86.08</td>
        <td>72.13</td>
        <td>88.25</td>
        <td>84.33</td>
    </tr>    
</table>

<sup>1 using Nvidia GTX 3070 Ti 8GB </sup><br>
<sup>2 (500 epochs, without early stopping)</sup><br>
<sup>* please note that <a href="https://github.com/robintibor/high-gamma-dataset">HGD</a> is for "executed movements" NOT "motor imagery"</sup>

##
This repository includes the implementation of the following attention schemes in the [*attention_models.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/attention_models.py) file: 

* [Multi-head self-attention (mha)](https://arxiv.org/abs/1706.03762)
* [Multi-head attention with locality self-attention (mhla)](https://arxiv.org/abs/2112.13492v1)
* [Squeeze-and-excitation attention (se)](https://arxiv.org/abs/1709.01507)
* [Convolutional block attention module (cbam)](https://arxiv.org/abs/1807.06521)
* [Efficient Channel Attention (eca)](https://arxiv.org/abs/1910.03151)
* [Coordinate Attention (ca)](https://arxiv.org/abs/1807.06521)

These attention blocks can be called using the *attention_block(net,  attention_model)* method in the [*attention_models.py*]file, where *'net'* is the input layer and *'attention_model'* indicates the type of the attention mechanism, which has five options: *None*, [*'mha'*](https://arxiv.org/abs/1706.03762), [*'mhla'*](https://arxiv.org/abs/2112.13492v1), [*'cbam'*](https://arxiv.org/abs/1807.06521), [*'eca'*](https://arxiv.org/abs/1910.03151), [*'ca'*](https://arxiv.org/abs/2103.029070),and [*'se'*](https://arxiv.org/abs/1709.01507).
```
Example: 
    input = Input(shape = (10, 100, 1))   
    block1 = Conv2D(1, (1, 10))(input)
    block2 = attention_block(block1,  'mha') # mha: multi-head self-attention
    output = Dense(4, activation="softmax")(Flatten()(block2))
```
##
The [*preprocess.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/preprocess.py) file loads and divides the dataset based on two approaches: 
1. [Subject-specific (subject-dependent)](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach. In this approach, we used the same training and testing data as the original [BCI-IV-2a](https://www.bbci.de/competition/iv/) competition division, i.e., trials in session 1 for training, and trials in session 2 for testing. 
2. [Leave One Subject Out (LOSO)](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach. LOSO is used for  **Subject-independent** evaluation. In LOSO, the model is trained and evaluated by several folds, equal to the number of subjects, and for each fold, one subject is used for evaluation and the others for training. The LOSO evaluation technique ensures that separate subjects (not visible in the training data) are used to evaluate the model.

The *get_data()* method in the [*preprocess.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/preprocess.py) file is used to load the dataset and split it into training and testing. This method uses the [subject-specific](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach by default. If you want to use the [subject-independent (LOSO)](https://link.springer.com/article/10.1007/s00521-021-06352-5#Sec9:~:text=Full%20size%20table-,Performance%20evaluation,-For%20the%20MI) approach, set the parameter *LOSO = True*.


## About DCNet
[DCNet](https://doi.org/10.1109/TII.2022.3197419) model consists of three main blocks: 

1. **Convolutional (CV) block**: The convolutional block consists of three convolutional layers, with a architecture similar to EEGNet. However, in contrast to EEGNet, EEG-DCNet introduces a 1 × 1 convolution layer in the depthwise convolution stage for dimensionality reduction. 
2. **Multi-Branch Parallel Atrous Convolution Block**: Through the hole convolution operation with different expansion rates, multi-scale information can be captured. These features from different scales are fused through 1×1 convolution.
3. **Sliding Window and SE Attention Block**: A sliding window strategy is used to divide the time series into multiple local subsequences. In each sliding window, the network uses a channel attention mechanism (SE module) to model the channel information of the feature map to improve the representation ability of effective features.
* [DCNet](https://doi.org/10.1109/TII.2022.3197419)  incorporates 1×1 convolutional layers and a multi-branch parallel atrous convolutional architecture to effectively augment the network’s capability to capture non-linear characteristics of EEG signals and improve the perception of multi-scale features

  


## Development environment
Models were trained and tested by a single GPU, Nvidia [GTX 3070Ti 8GB](https://www.nvidia.com/en-me/geforce/graphics-cards/rtx-2070/) (Driver Version: [512.78](https://www.nvidia.com/download/driverResults.aspx/188599/en-us/), [CUDA 11.3](https://developer.nvidia.com/cuda-11.3.0-download-archive)), using Python 3.9 with [TensorFlow](https://www.tensorflow.org/) framework. [Anaconda 3](https://www.anaconda.com/products/distribution) was used on [Ubuntu 20.04.4 LTS](https://releases.ubuntu.com/20.04/) and [Windows 11](https://www.microsoft.com/en-hk/software-download/windows11).
The following packages are required:

* TensorFlow 2.7
* matplotlib 3.5
* NumPy 1.20
* scikit-learn 1.0
* SciPy 1.7

## Dataset 
The [BCI Competition IV-2a](https://www.bbci.de/competition/iv/#dataset2a) dataset needs to be downloaded, and the data path should be set in the 'data_path' variable in the [*main_TrainValTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainValTest.py) file. The dataset can be downloaded from [here](http://bnci-horizon-2020.eu/database/data-sets).

The [BCI Competition IV-2b](https://www.bbci.de/competition/iv/#dataset2b) dataset needs to be downloaded, and the data path should be set in the 'data_path' variable in the [*main_TrainValTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainValTest.py) file. The dataset can be downloaded from [here](http://bnci-horizon-2020.eu/database/data-sets).

The [high-gamma-dataset](https://github.com/robintibor/high-gamma-dataset) dataset needs to be downloaded, and the data path should be set in the 'data_path' variable in the [*main_TrainValTest.py*](https://github.com/Altaheri/EEG-ATCNet/blob/main/main_TrainValTest.py) file. The dataset can be downloaded from [here](https://github.com/robintibor/high-gamma-dataset).
