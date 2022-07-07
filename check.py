import os


def check_is_repeat(json_dir):
    files_name = []
    for root, dirs, names in os.walk(json_dir):
        files_name.extend(names)
    if len(files_name) == len(set(files_name)):
        print("文件不重复")
    else:
        print("文件重复")


if __name__ == '__main__':
    json_dir = r"D:\work\dataSet\organ-segmentation\v1\processed_json"
    check_is_repeat(json_dir)
