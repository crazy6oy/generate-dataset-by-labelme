import os
import cv2
import tqdm

import utils


def get_dataset_include_surgery(files_path, categories, testset_ratio, shape_type, divide_testset):
    sample_num, sample = utils.collect_label_number(
        files_path, shape_type)
    return utils.divide_dataset(sample_num, sample, categories, testset_ratio,
                                divide_testset)


if __name__ == '__main__':
    json_dir = r"D:\work\dataSet\organ-segmentation\v1\processed_json\v3"
    dataset_save_dir = r"D:\work\dataSet\organ-segmentation\v1\dataset"
    log_path = os.path.join(dataset_save_dir, "log.log")
    categories = ('cystic plate',
                  'dissected windows in the hepatocystic triangle',
                  'cystic artery',
                  'cystic duct',
                  'gallbladder')
    categories_id = {'cystic artery': 1,
                     'cystic duct': 2,
                     'cystic plate': 3,
                     'dissected windows in the hepatocystic triangle': 4,
                     'gallbladder': 5}
    testset_ratio = 1 / 10
    shape_type = "polygon"
    divide_testset = True
    is_save = True
    is_obliterated_data = True
    os.makedirs(dataset_save_dir, exist_ok=True)

    files_path = utils.format_label_file(json_dir)
    divide_results, divide_number = get_dataset_include_surgery(files_path, categories, testset_ratio, shape_type,
                                                                divide_testset)
    print(divide_number)
    if not is_save:
        exit()

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'-' * 64}\n{json_dir}\n")
        f.write(f"{divide_results}\n{divide_number}\n{categories_id}\n")
        f.flush()

        for surgery_name in files_path.keys():
            dataset_name = [x for x in divide_results.keys(
            ) if surgery_name in divide_results[x]]
            if len(dataset_name) != 1:
                raise RuntimeError("数据集划分错误")
            dataset_name = dataset_name[0]
            for json_path in tqdm.tqdm(files_path[surgery_name], desc=surgery_name):
                json_path = json_path[0]
                image, mask = utils.make_segmentation_data(
                    json_path, categories_id, is_obliterated_data)
                image_path = os.path.join(dataset_save_dir, dataset_name, "image",
                                          os.path.split(json_path)[-1].replace(".json", ".jpg"))
                mask_path = os.path.join(dataset_save_dir, dataset_name, "mask",
                                         os.path.split(json_path)[-1].replace(".json", ".png"))
                os.makedirs(os.path.split(image_path)[0], exist_ok=True)
                os.makedirs(os.path.split(mask_path)[0], exist_ok=True)
                cv2.imwrite(image_path, image)
                cv2.imwrite(mask_path, mask)
                f.write(f"{image_path}\n")
                f.write(f"{mask_path}\n")
                f.flush()
