import  os



#获取数据集中的图像和标签，并将其组合为一个列表，列表中的每个元素为一个元组（image_path,label_path）
def get_image_label(path):
    image_path = os.path.join(path, 'Freeman')
    label_path = os.path.join(path, 'label')
    if not os.path.exists(image_path):
        print("image not exist")
        raise FileNotFoundError
    if not os.path.exists(label_path):
        print("label not exist")
        raise FileNotFoundError
    image_list = [os.path.join(image_path, image) for image in os.listdir(image_path)]
    label_list = [os.path.join(label_path, label) for label in os.listdir(label_path)]
    image_label = []
    for i in range(len(image_list)):
        image_label.append((image_list[i],label_list[i]))
    return image_label


def get_image_scatter(path):
    image_path = os.path.join(path, 'image')
    label_path = os.path.join(path, 'scatter')
    if not os.path.exists(image_path):
        print("image not exist")
        raise FileNotFoundError
    if not os.path.exists(label_path):
        print("scatter not exist")
        raise FileNotFoundError
    image_list = [os.path.join(image_path, image) for image in os.listdir(image_path)]
    label_list = [os.path.join(label_path, label) for label in os.listdir(label_path)]
    image_label = []
    for i in range(len(image_list)):
        image_label.append((image_list[i], label_list[i]))
    return image_label

def get_HH_label(path):
    image_path = os.path.join(path, 'HH')
    label_path = os.path.join(path, 'label')
    if not os.path.exists(image_path):
        print("HH not exist")
        raise FileNotFoundError
    if not os.path.exists(label_path):
        print("scatter not exist")
        raise FileNotFoundError
    image_list = [os.path.join(image_path, image) for image in os.listdir(image_path)]
    label_list = [os.path.join(label_path, label) for label in os.listdir(label_path)]
    image_label = []
    for i in range(len(image_list)):
        image_label.append((image_list[i], label_list[i]))
    return image_label


def get_image_label_scatter(path):
    image_path = os.path.join(path, 'Freeman')
    label_path = os.path.join(path, 'label')
    scatter_path = os.path.join(path, 'scatter')
    if not os.path.exists(image_path):
        print("image not exist")
        raise FileNotFoundError
    if not os.path.exists(label_path):
        print("label not exist")
        raise FileNotFoundError
    if not os.path.exists(scatter_path):
        print("scatter not exist")
        raise FileNotFoundError
    image_list = [os.path.join(image_path, image) for image in os.listdir(image_path)]
    label_list = [os.path.join(label_path, label) for label in os.listdir(label_path)]
    scatter_list = [os.path.join(scatter_path, scatter) for scatter in os.listdir(scatter_path)]
    image_label_scatter = []
    for i in range(len(image_list)):
        image_label_scatter.append((image_list[i], label_list[i],scatter_list[i]))
    return image_label_scatter


def get_all_path(path):
    image_path = os.path.join(path, 'Freeman')
    HH_path = os.path.join(path, 'HH')
    HV_path = os.path.join(path, 'HV')
    VH_path = os.path.join(path, 'VH')
    VV_path = os.path.join(path, 'VV')
    label_path = os.path.join(path, 'label')
    if not os.path.exists(image_path):
        print("image not exist")
        raise FileNotFoundError
    if not os.path.exists(HH_path):
        print("HH not exist")
        raise FileNotFoundError
    if not os.path.exists(HV_path):
        print("HV not exist")
        raise FileNotFoundError
    if not os.path.exists(VH_path):
        print("VH not exist")
        raise FileNotFoundError
    if not os.path.exists(VV_path):
        print("VV not exist")
        raise FileNotFoundError
    if not os.path.exists(label_path):
        print("label not exist")
        raise FileNotFoundError
    image_list = [os.path.join(image_path, image) for image in os.listdir(image_path)]
    HH_list = [os.path.join(HH_path, HH) for HH in os.listdir(HH_path)]
    HV_list = [os.path.join(HV_path, HV) for HV in os.listdir(HV_path)]
    VH_list = [os.path.join(VH_path, VH) for VH in os.listdir(VH_path)]
    VV_list = [os.path.join(VV_path, VV) for VV in os.listdir(VV_path)]
    label_list = [os.path.join(label_path, label) for label in os.listdir(label_path)]
    all_path = []
    for i in range(len(image_list)):
        all_path.append((image_list[i], HH_list[i], HV_list[i], VH_list[i], VV_list[i], label_list[i]))
    return all_path
