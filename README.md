## This is a pytorch implementation of the paper *[Zhen Zhang*, Yanyu Wang, Xingxin Ruan, Xiangyu Zhang. Lithium-ion batteries lifetime early prediction using domain adversarial learning. Renewable and sustainable energy reviews. 208，115035, 2025.]*，请随便学习使用，如有发表文章请引用该文章

##MIT 两个数据集的版权属于原作者，本代码只是把它部分内容提取出来方便使用，如有发表请引用MIT的文献：
[1]	Severson K A, Attia P M, Jin N, Perkins N, Jiang B, Yang Z, Chen M H, Aykol M, Herring P K, Fraggedakis D, Bazant M Z, Harris S J, Chueh W C, Braatz R D. Data-driven prediction of battery cycle life before capacity degradation. Nature Energy 2019; 4: 383-391.
[2]  Attia P M, Grover A, Jin N, Severson K A, Markov T M, Liao Y H, Chen M H, Cheong B, Perkins N, Yang Z, Herring P K, Aykol M, Harris S J, Braatz R D, Ermon S, Chueh W C. Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature 2020; 578(7795): 397-402.

#### Environment
- Pytorch 
- Python 

#### files Structure

```
--root
	|--data: MIT 124和45两个数据集的数据，经过处理只保留了程序使用的内容，如有侵权，请告知！
		|--deltaq124.mat 				MIT Dataset1的数据文件
		|--eol124.mat 				MIT Dataset1的寿命文件
		|--deltaq45.mat 				MIT Dataset2的数据文件
		|--eol45.mat 					MIT Dataset2的寿命文件     
        |--model_dann.py				DANN模型
        |--CNN_fix_124_45_dann_train.py	训练文件
	|--CNN_fix_45_dann_test.py:		测试文件
        |--functions.py:					梯度反转函数
        |--tsne_features.py:				特征可视化

