import json

def read_imagenet_classes(file_path: str) -> dict[int, str]:
    with open(file_path, "r") as f:
        class_list = json.load(f)
        index_to_class_name = {index:class_name.title() for index, class_name in enumerate(class_list)}
    
    return index_to_class_name