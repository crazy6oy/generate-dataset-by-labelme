# generate-dataset-by-labelme

check.py支持：

1. 一个文件夹下是否存在同名文件（防止多个标注任务有标重的情况）；

utils.py支持:

1. 划分数据集的工具函数;
2. 生成数据的工具函数;

label_json_tools.py支持:

1. 数据清理（修改标签名称）;
2. 数据统计（出现统计）;
3. 生成json中编码的图像;

make_segmentation_dataset.py特点:

1. 只划分分割数据集;
2. 根据categories里面的类别顺序划分数据集（数量不均衡把难划分的/少的数据放在前面）;
3. 根据categories_id定制类别ID映射表；
4. 根据divide_testset可以选择是否划分数据集;

