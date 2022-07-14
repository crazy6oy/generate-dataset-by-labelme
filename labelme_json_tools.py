import os
import cv2
import json
import tqdm
from prettytable import PrettyTable

from utils import str_to_image


def json_to_image(json_path):
    with open(json_path, encoding="utf-8") as f:
        json_message = json.load(f)

    return str_to_image(json_message["imageData"])


def draw_all_json(json_dir, save_dir=None):
    for root, dirs, names in os.walk(json_dir):
        for name in tqdm.tqdm(names, desc=os.path.split(root)[-1]):
            if name.endswith(".json"):
                json_path = os.path.join(root, name)
                image = json_to_image(json_path)

                image_path = json_path.replace(".json", ".jpg")
                if save_dir is not None:
                    image_path = image_path.replace(json_dir, save_dir)
                    os.makedirs(os.path.split(image_path)[0], exist_ok=True)
                cv2.imwrite(image_path, image)


def statistic_semantic_num(json_dir, connector="_", shape_type="polygon"):
    print("semantic statistic start!")
    output = {}

    for root, dirs, names in os.walk(json_dir):
        for name in tqdm.tqdm(names, desc=os.path.split(root)[-1]):
            prefix = name.split(connector)[0]
            if prefix not in output.keys():
                output[prefix] = {}

            json_path = os.path.join(root, name)
            with open(json_path, encoding="utf-8") as f:
                json_message = json.load(f)
            file_appear_class = []
            for shape in json_message["shapes"]:
                if shape["shape_type"] != shape_type:
                    continue
                file_appear_class.append(shape["label"])
            for appear_class in sorted(set(file_appear_class)):
                if appear_class not in output[prefix].keys():
                    output[prefix][appear_class] = 0
                output[prefix][appear_class] += 1

    print("semantic statistic finish!")
    return output


def statistic_region_num(json_dir, shape_type="polygon"):
    outputs = {}
    for root, dirs, names in os.walk(json_dir):
        for name in tqdm.tqdm(names, desc=os.path.split(root)[-1]):
            if name.split(".")[-1] != "json":
                continue
            json_path = os.path.join(root, name)

            with open(json_path, encoding="utf-8") as f:
                json_msg = json.load(f)

            for msg in json_msg["shapes"]:
                if msg["shape_type"] != shape_type:
                    continue

                if msg["label"] not in outputs.keys():
                    outputs[msg["label"]] = 0
                outputs[msg["label"]] += 1

    print(outputs)


def merge_dict(*dicts):
    keys = []
    for single_dict in dicts:
        for key in single_dict.keys():
            keys.append(key)
    keys = sorted(set(keys))

    output = {}
    for key in keys:
        output[key] = []

    for single_dict in dicts:
        for key in single_dict.keys():
            output[key].extend(single_dict[key])

    return output


def fix_labels(json_dir, save_dir, fix_map, continue_labels, shape_type="polygon"):
    json_files_name = [x for x in os.listdir(
        json_dir) if x.split(".")[-1] == "json"]
    os.makedirs(save_dir, exist_ok=True)

    for json_name in tqdm.tqdm(json_files_name):
        json_path = os.path.join(json_dir, json_name)

        with open(json_path, encoding="utf-8") as f:
            json_msg = json.load(f)

        if json_msg["imageData"] is None:
            continue

        processed_shapes = []
        for msg in json_msg["shapes"]:
            if msg["shape_type"] != shape_type:
                continue

            if msg["label"] in continue_labels:
                continue

            msg["label"] = fix_map[msg["label"]]
            processed_shapes.append(msg)
        json_msg["shapes"] = processed_shapes

        save_json_path = os.path.join(save_dir, json_name)
        with open(save_json_path, "w", encoding="utf-8") as f:
            json.dump(json_msg, f, ensure_ascii=False, indent=2)


def statistic_appear_num_in_every_dataset(json_dir, divide_log):
    semantic_statistic_results = statistic_semantic_num(json_dir)
    with open(divide_log, encoding="utf-8") as f:
        json_message = json.load(f)
    divide_collect = merge_dict(*json_message.values())
    divide_num = {}
    for key in semantic_statistic_results.keys():
        dataset_name = [k for k in divide_collect.keys()
                        if key in divide_collect[k]]
        if len(dataset_name) != 1:
            print("{} in {}".format(key, dataset_name))
            continue
        else:
            dataset_name = dataset_name[0]

        if dataset_name not in divide_num.keys():
            divide_num[dataset_name] = {}
        for class_name in semantic_statistic_results[key].keys():
            if class_name not in divide_num[dataset_name].keys():
                divide_num[dataset_name][class_name] = 0
            divide_num[dataset_name][class_name] += semantic_statistic_results[key][class_name]

    table_title = ["class name", "train", "valid", "test"]
    table = PrettyTable(table_title)
    for class_name in sorted(divide_num["train"].keys()):
        table.add_row([class_name,
                       divide_num["train"][class_name],
                       divide_num["valid"][class_name],
                       divide_num["test"][class_name]])
    print(table)


if __name__ == '__main__':
    fix_map = {
        'Dissected windows in the \rhepatocystic triangle': 'dissected windows in the hepatocystic triangle',
        'Cystic artery': 'cystic artery',
        'Cystic duct': 'cystic duct',
        'Gallbladder': 'gallbladder',
        'Liver': 'liver',
        'Cystic plate': 'cystic plate',
        'ignore': 'unknow',
    }
    continue_labels = [
        '_background_',
        'Absorbable Clip_1',
        'Absorbable Clip_2',
        'Absorbable Clip_3',
        'Absorbable Clip_4',
        'Absorbable Clip_5',
        'Absorbable Clip_6',
        'Aspirator_1',
        'Atraumatic fixation forceps shafts_1',
        'Atraumatic fixation forceps tip_1',
        'Atraumatic forceps shafts_1',
        'Atraumatic forceps tip_1',
        'Cautery hook shafts_1',
        'Cautery hook tip_1',
        'Claw grasper tip_1',
        'Clip applier shafts_1',
        'Clip applier tip_1',
        'Coagulator shafts_1',
        'Coagulator tip_1',
        'Dissecting forceps tip_1',
        'excess',
        'Gauze_1',
        'Gauze_2',
        'Maryland dissecting forceps shafts_1',
        'Maryland dissecting forceps tip_1',
        'Metal Clip_1',
        'Metal Clip_2',
        'Rouviere sulcus',
        'Scissor shafts_1',
        'Scissor tip_1',
        'Specimen bag_1',
        'Straight dissecting forceps shafts_1',
        'Straight dissecting forceps tip_1',
        'Trocar_1',
        'Trocar_2',
        "Atrumatic forceps tip",
        "Maryland dissecting forceps tip"
    ]

    json_dir = r"D:\work\dataSet\organ-segmentation\v1\origin_json\v3"
    save_dir = r"Z:\withai\dataset\origin-data\lc-instruments-segmentation\20-categories"
    divide_log = r"D:\work\dataSet\organ-segmentation\v3\divide_log.json"

    # 清洗标签
    # fix_labels(json_dir, save_dir, fix_map, continue_labels)
    statistic_region_num(save_dir)

    # 统计各数据集中各类别出现次数
    # statistic_appear_num_in_every_dataset(json_dir,divide_log)

    # 生成json中的图片
    # draw_all_json(json_dir)
