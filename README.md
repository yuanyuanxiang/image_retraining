# image_retraining
迁移学习Inception v3。代码来自：https://github.com/tensorflow/models/research/syntaxnet/tensorflow/tensorflow/examples

迁移学习Inception V3

1、	资料来源

Tensorflow：models\research\syntaxnet\tensorflow\tensorflow\examples\image_retraining

2、	迁移学习

cd到image_retraining目录，然后执行

python  retrain.py  --image_dir  IMAGE_DIR  --model_dir  MODEL_DIR

如：python  retrain.py  --image_dir  ..\flower_photos  --model_dir  ..\inception-2015-12-05

在inception-2015-12-05目录下，inception-2015-12-05.tgz文件必须存在，否则会联网下载。

通过输入 "python  retrain.py  --help" 查看参数说明。
 
3、	结果保存

迁移学习的结果保存在tmp目录，如果在D盘，文件保存在D:\tmp。
 
4、	模型调用

python  label_image.py  --image  IMAGE_FILE  --graph  GRAPH  --labels  LABELS

如：

python  label_image.py  --image  ..\flower_photos\daisy\5547758_eea9edfd54_n.jpg  --graph  \tmp\output_graph.pb  --labels  \tmp\output_labels.txt
 
预测结果及未迁移学习的结果如DOC文档所示。
 
2018年6月4日
