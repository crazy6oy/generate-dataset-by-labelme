import os
import cv2
import json
import tqdm
import base64
import random
from math import ceil
import numpy as np


# --------------------------------------- common -----------------------------------------
def str_to_image(str):
    """
    ascii编码字符串转换为图像（opencv直接使用，ndarray类型）
    :param str: ascii编码字符串的图片信息
    :return: BGR通道顺序的图片，维度为HWC
    """
    image_str = str.encode('ascii')
    image_byte = base64.b64decode(image_str)

    img = cv2.imdecode(np.asarray(bytearray(image_byte),
                       dtype='uint8'), cv2.IMREAD_COLOR)  # for jpg
    return img


def format_label_file(labelFoder) -> dict:
    """
    文件夹下的文件'手术名称_时间.xxx'，转换为以手术为单位的图像汇总。
    :param labelFoder: 需要涉及的文件目录
    :return: 返回一个以手术名称为key，这个手术所包含的图像名称为value的字典。
    """
    files_path = {}
    for root, dirs, names in os.walk(labelFoder):
        if names == []:
            continue
        for file_name in names:
            surgery_name = file_name.split("_")[0]
            file_path = os.path.join(root, file_name)
            if surgery_name not in files_path.keys():
                files_path[surgery_name] = []
            files_path[surgery_name].append(file_path)
    for surgery_name in files_path.keys():
        files_path[surgery_name] = [[x, ]
                                    for x in sorted(files_path[surgery_name])]

    return files_path


def divide_dataset(sample_num, sample, categories, testset_ratio, divide_testset):
    """
    根据数量划分数据集
    :param sample_num: 手术名称作为键，类别和数量的映射关系作为值的dict。
    :param sample: 类别作为键，样本数量作为值的dict。
    :param categories: 划分数据集时划分类别的顺序，必须涉及所有类别。
    :param testset_ratio: 测试集、验证集占比。
    :param divide_testset: 是否划分测试集。
    :return: 各个手术属于哪个数据集以及每个数据集数据分布情况。
    """
    divide_results = {"train": [], "valid": [], "test": []}
    divide_number = {"train": {}, "valid": {}, "test": {}}
    rule_sample_num = sample_num.copy()
    for category in categories:
        if category not in divide_number["train"].keys():
            divide_number["train"][category] = 0
        if category not in divide_number["valid"].keys():
            divide_number["valid"][category] = 0
        if category not in divide_number["test"].keys():
            divide_number["test"][category] = 0

        choise_surgery_name = [
            x for x in rule_sample_num if category in rule_sample_num[x]]
        random.shuffle(choise_surgery_name)
        for surgery_name in choise_surgery_name:
            if category not in rule_sample_num[surgery_name]:
                continue
            if rule_sample_num[surgery_name][category] > sample[category] * testset_ratio:
                dataset = "train"
            else:
                if divide_testset and (divide_number["test"][category] + rule_sample_num[surgery_name][category]) < \
                        sample[category] * testset_ratio:
                    dataset = "test"
                elif (divide_number["valid"][category] + rule_sample_num[surgery_name][category]) < sample[
                        category] * testset_ratio:
                    dataset = "valid"
                else:
                    dataset = "train"

            divide_results[dataset].append(surgery_name)
            for category_name in rule_sample_num[surgery_name].keys():
                if category_name not in divide_number[dataset].keys():
                    divide_number[dataset][category_name] = 0
                divide_number[dataset][category_name] += rule_sample_num[surgery_name][category_name]
            rule_sample_num.pop(surgery_name)
    return divide_results, divide_number


# ---------------------------------------- Image ------------------------------------------
def collect_label_number(files_path, type_shape="polygon"):
    """
    统计区域个数

    :param files_path: 视频/大类为key, json文件list为values;
    :param type_shape: 统计的标注类型[polygon、rectangle...];
    :return 以视频为最小单位和以图片为最小单位统计的各类别数量.
    """

    outputs = {}
    all_sample = {}
    for surgery_name in files_path.keys():
        if surgery_name not in outputs.keys():
            outputs[surgery_name] = {}
        for i in range(len(files_path[surgery_name])):
            json_path = files_path[surgery_name][i][0]
            with open(json_path, encoding="utf-8") as f:
                label_message = json.load(f)
            for shape in label_message["shapes"]:
                if shape["shape_type"] != type_shape:
                    continue
                label_name = shape["label"]
                if label_name not in outputs[surgery_name].keys():
                    outputs[surgery_name][label_name] = 0
                if label_name not in all_sample.keys():
                    all_sample[label_name] = 0
                outputs[surgery_name][label_name] += 1
                all_sample[label_name] += 1

    return outputs, all_sample


def get_image_color_histogram(image):
    """
    获取单帧图像BGR各通道色彩值直方图/值
    根据formula_mode决定返回图和各通道计算结果值

    :param image:类型（np.array-uint8）计算用图，BGR通道顺序
    :return RGB各通道像素分布
    """
    channel_sorted = ["R", "G", "B"]
    reture_value = {}

    h, w, _ = image.shape
    for channel_id in range(3):
        color_values_statistic = np.unique(
            image[..., channel_id], return_counts=True)
        channel_pixel = [0] * 256
        for i in range(color_values_statistic[0].shape[0]):
            color_values = color_values_statistic[0][i]
            color_values_percent = color_values_statistic[1][i] / (h * w)

            channel_pixel[color_values] = color_values_percent
        reture_value[channel_sorted[channel_id]] = channel_pixel

    return reture_value


def make_segmentation_data(json_path, target_categories, is_obliterated_data, type_shape="polygon"):
    """
    生成原图和分割标签mask

    :param json_path: json路径;
    :param target_categories: 需要生成标签的ID映射表;
    :param is_obliterated_data: 是否有去除的数据;
    :param type_shape: 生成标签的类别;
    :return 原图和mask
    """
    with open(json_path, encoding="utf-8") as f:
        label_msg = json.load(f)
        f.close()
    img = str_to_image(label_msg["imageData"])
    H, W, _ = img.shape
    mask = np.zeros_like(img, dtype=np.uint8)
    for set_msg in label_msg["shapes"]:
        if set_msg["shape_type"] != type_shape:
            continue
        label = set_msg["label"].lower()
        if label not in target_categories.keys():
            continue
        label_id = target_categories[label]
        points = set_msg["points"]
        pts = [[min(max(0, x), W), min(max(0, y), H)] for x, y in points]
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts], (label_id, label_id, label_id))

    # unknow
    if is_obliterated_data:
        for set_msg in label_msg["shapes"]:
            if set_msg["label"] == "unknow":
                points = set_msg["points"]
                pts = [[min(max(0, x), W), min(max(0, y), H)]
                       for x, y in points]
                pts = np.array(pts, dtype=np.int32)
                cv2.fillPoly(mask, [pts], (0, 0, 0))
                cv2.fillPoly(img, [pts], (0, 0, 0))

    return img, mask


# --------------------------------------- Sequence -----------------------------------------
def calculate_appear_count(score_same, score_different, choise_category):
    """
    计算某个分数出现次数，分数列表2中每出现一次算0.5次
    :param score_same: 分数列表1
    :param score_different: 分数列表2
    :param choise_category: 需要的类别
    :return: 返回出现次数
    """
    same_appear_count = sum([1 for x in score_same if x == choise_category])
    different_appear_count = sum(
        [1 for x in score_different if x == choise_category]) * 0.5
    appear_count = same_appear_count + different_appear_count

    return appear_count


def fix_images_list(input):
    """
    图像名称不连续时填充
    :param input: 手术名称为key，图片名称为value
    :return: 和input相同只不过添加了确实图片。
    """
    for surgery_name in input.keys():
        paths = [x[0] for x in input[surgery_name]]
        path_example = input[surgery_name][0][0]
        ids = [int(os.path.split(x[0])[-1].split(".")[0].split("_")[-1])
               for x in input[surgery_name]]
        max_id = max(ids)
        min_id = min(ids)

        for i in range(max(0, min_id - 1), max_id + 2):
            format_str = "{}_{:0>5}.{}".format(
                path_example.split('_')[0], i, path_example.split('.')[-1])
            if format_str not in paths:
                input[surgery_name].append([format_str, [None, None, None]])
        input[surgery_name] = sorted(input[surgery_name], key=lambda x: x[0])
    return input


def statistic_lc10000_sequence_label_second(label_files_folder):
    """

    :param label_files_folder:
    :return:
    """
    """
    :param sample_num: 手术名称作为键，类别和数量的映射关系作为值的dict。
    :param sample: 类别作为键，样本作为值的dict。
    """
    output_sample_num = {}
    output_sample = {}
    for (root, dirs, names) in os.walk(label_files_folder):
        if names == []:
            continue
        for name in names:
            file_path = os.path.join(root, name)
            surgery_name = name[:name.rfind(".")]
            if surgery_name not in output_sample_num.keys():
                output_sample_num[surgery_name] = {}
            with open(file_path, encoding="utf-8") as f:
                label_message = json.load(f)
            for ob in label_message:
                start = ob["start"]
                end = ob["end"]
                label_name = ob["label"]
                if label_name not in output_sample_num[surgery_name].keys():
                    output_sample_num[surgery_name][label_name] = 0
                if label_name not in output_sample.keys():
                    output_sample[label_name] = []
                sample = ["{}_{:0>2}.{:0>2}.{:0>2}.jpg".format(surgery_name, x // 3600, x % 3600 // 60, x % 60) for x in
                          range(int(start), ceil(end) + 1)]

                output_sample[label_name].extend(sample)
                output_sample_num[surgery_name][label_name] += len(sample)
    return output_sample_num, output_sample


def statistic_sample_num(sample_dir):
    outputs = {}
    for root, dirs, names in os.walk(sample_dir):
        if names == []:
            continue
        for name in names:
            surgery_name = name.split("_")[0]
            if surgery_name not in outputs:
                outputs[surgery_name] = []
            outputs[surgery_name].append(name)
    for surgery_name in outputs.keys():
        outputs[surgery_name] = sorted(outputs[surgery_name])
    return outputs
