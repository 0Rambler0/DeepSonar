# DeepSonar复现
​	该项目为MM 2020发表的文章《DeepSonar: Towards Effective and Robust Detection of AI-Synthesized Fake Voices》的复现代码。

### 相关依赖
- [Python 2.7.15](https://www.continuum.io/downloads)

- [Keras 2.2.4](https://keras.io/)

- [Tensorflow 1.8.0](https://www.tensorflow.org/)

- CUDA 9 

- See requirements.txt
### 数据
​	该项目采用的数据集为FOR（fake or real）数据集，下载地址：http://bil.eecs.yorku.ca/aptly-lab.

### 使用 

​	1.运行`get_net_param.py`生成数据集参数文件， 记得修改103行和104行的路径

​	2.运行`create_raw_data.py`生成原始特征数据文件，同样需要修改里面的路径，生成的原始特征文件较大，14000条音频生成后大概有28g，记得保证硬盘空间足够。

​	3.运行`evaluate.py`，还是需要修改路径。

​	注：评估数据集的组织形式应参照for数据集，所有子数据集放在一个目录下，每个子数据集中的fake和real音频分别放在fake、real目录下


​    




