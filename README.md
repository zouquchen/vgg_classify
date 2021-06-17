# 自定义数据集
1. 下载数据集：https://www.kaggle.com/ashishsaxena2209/animal-image-datasetdog-cat-and-panda
2. 将数据集按照9:1的比例分成训练集train和测试集test，分别存在train和test文件夹下。

# 训练模型
1. 打开vgg_train.py文件，修改数据集的路径对应自定义数据集中train和test的路径。
2. 运行程序开始训练，不需要使用预训练模型，epoch达到40（epochs*milestone[1]）轮时开始保存模型。

# 测试单张图片
1. 打开vgg_classify.py文件，修改模型路径和所需检测图片的路径。
2. 运行程序得出检测结果。

# 可视化界面
1. 打开run.py文件，修改27行模型的路径。
2. 运行程序即可打开系统界面。
