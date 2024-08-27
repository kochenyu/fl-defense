import copy

import torch.nn.functional as F
import torch
import os
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def getL2TensorCosineSimilary(midData, allData, result):
    normalized_vector1 = F.normalize(midData, p=2, dim=1)
    for key, data in allData.items():
        # L2归一化
        normalized_vector2 = F.normalize(data, p=2, dim=1)
        # 计算余弦相似度
        similarity = F.cosine_similarity(normalized_vector1, normalized_vector2)

        # 将得到的余弦相似度的数组归一
        t = torch.min(similarity)

        # 打印结果
        result[key] = t
        # print("余弦相似度:", similarity)


def check_threshold(result, threshold):
    # 初始化状态码为False
    well_code = True

    # 遍历字典中的值
    for value in result.values():
        # 如果值小于阈值，则将状态码更新为True，并跳出循环
        if value < threshold:
            well_code = False
            break

    return well_code


def get_Min_cosvalue(result):
    l = list(result.values())
    s = torch.stack(l)
    min = torch.min(s)
    return min


def getPearson(base, title):
    # 将张量展平为一维张量
    flat_tensor1 = base.view(-1)
    flat_tensor2 = title.view(-1)

    # 连接两个一维张量
    concatenated_tensor = torch.cat((flat_tensor1, flat_tensor2), dim=0)

    # 计算 Pearson 相关系数
    corr = torch.corrcoef(concatenated_tensor)
    return corr


def getSeverWeightValue(base, title):
    # print("---服务器聚集的权重比较------")
    # print(base)
    # print(title)
    # t = F.cosine_similarity(base, title, dim=0)
    normalized_vector1 = F.normalize(base, p=2, dim=1)
    normalized_vector2 = F.normalize(title, p=2, dim=1)
    similarity = F.cosine_similarity(normalized_vector1, normalized_vector2)
    t = torch.mean(similarity)
    return t


# equal_to_value = torch.eq(label, 9)
# if torch.any(equal_to_value):
#     print("断点")

def flip_labels(label):
  return torch.full_like(label, 0)

# def flip_labels(label):
#     M = 10
#     modified_label = []
#
#     for l in label:
#         k = M - l - 1
#         modified_label.append(k)
#     modified_label_tensor = torch.tensor(modified_label)
#
#     return modified_label_tensor
    # 在 DataLoader 中对标签进行翻转


def flip_labels_in_loader(loader):
    for batch in loader:
        data, label = batch
        yield data, flip_labels(label)


def agg_midden_value(agg_client_dict):
    # agg_client_dict 为{客户端id: 梯度字典}
    # 新建一个字典，用于存储每个梯度键的梯度列表
    all_grads_dict = {}
    # 遍历agg_client_dict中的每个客户端的梯度信息
    # 遍历agg_client_dict中的每个客户端的梯度信息
    for client_id, gradients in agg_client_dict.items():
        # 遍历每个客户端的梯度信息，将每个梯度键的梯度存储到all_grads_dict中
        for grad_key, grad_value in gradients.items():
            if grad_key not in all_grads_dict:
                all_grads_dict[grad_key] = []
            # 将梯度张量移动到相同的设备上
            if not torch.is_tensor(grad_value):
                grad_value = torch.from_numpy(grad_value)
            all_grads_dict[grad_key].append(grad_value)

    # grad_datas = []
    # for key, value in all_grads_dict.items():
    #     if 'fc2.weight' in key:
    #         for v in value:
    #             grad_datas.append(v.grad)
    #
    # all_grads_dict['fc2.weight.grad'] = grad_datas
    # 新建一个字典，用于存储每个梯度键的中位数梯度
    median_grads_dict = {}

    # 遍历all_grads_dict中的每个梯度键，计算中位数梯度
    for grad_key, grad_list in all_grads_dict.items():
        # 将梯度列表合并为一个张量
        # grad_list = grad_list.to(device)
        stacked_grads = torch.stack(grad_list).float()
        # 计算中位数梯度
        median_grads = torch.median(stacked_grads, dim=0).values
        # 存储中位数梯度
        median_grads_dict[grad_key] = median_grads
    return median_grads_dict


def agg_calc_mean_value(agg_client_dict):
    # agg_client_dict 为{客户端id: 梯度字典}
    # 新建一个字典，用于存储每个梯度键的梯度列表
    all_grads_dict = {}
    # 遍历agg_client_dict中的每个客户端的梯度信息
    for client_id, gradients in agg_client_dict.items():
        # 遍历每个客户端的梯度信息，将每个梯度键的梯度存储到all_grads_dict中
        for grad_key, grad_value in gradients.items():
            if grad_key not in all_grads_dict:
                all_grads_dict[grad_key] = []
            # 将梯度张量移动到相同的设备上
            if not torch.is_tensor(grad_value):
                grad_value = torch.from_numpy(grad_value)
            all_grads_dict[grad_key].append(grad_value)

    mean_grads_dict = {}

    # 遍历all_grads_dict中的每个梯度键，计算均值梯度
    for grad_key, grad_list in all_grads_dict.items():
        # 检查张量形状是否一致，如果不一致，进行调整
        # max_shape = max([grad.shape for grad in grad_list])
        # grad_list = [torch.nn.functional.pad(grad, (0, max_shape[0] - grad.shape[0])) for grad in grad_list]
        # 将梯度列表合并为一个张量，并转换为浮点数类型
        stacked_grads = torch.stack(grad_list).float()  # 将张量转换为浮点数类型
        # 计算均值
        mean_grads = torch.mean(stacked_grads, dim=0)
        # 存储均值梯度
        mean_grads_dict[grad_key] = mean_grads
    return mean_grads_dict

def agg_mean_value(agg_client_dict):
    # agg_client_dict 为{客户端id: 梯度字典}
    # 新建一个字典，用于存储每个梯度键的梯度列表
    all_grads_dict = {}
    # 遍历agg_client_dict中的每个客户端的梯度信息
    for client_id, gradients in agg_client_dict.items():
        # 遍历每个客户端的梯度信息，将每个梯度键的梯度存储到all_grads_dict中
        for grad_key, grad_value in gradients.items():
            if grad_key not in all_grads_dict:
                all_grads_dict[grad_key] = []
            # 将梯度张量移动到相同的设备上
            if not torch.is_tensor(grad_value):
                grad_value = torch.from_numpy(grad_value)
            all_grads_dict[grad_key].append(grad_value)

    median_grads_dict = {}

    # 遍历all_grads_dict中的每个梯度键，计算中位数梯度
    for grad_key, grad_list in all_grads_dict.items():
        # 将梯度列表合并为一个张量
        stacked_grads = torch.stack(grad_list).float()
        # 排序并去掉最大值与最小值
        sorted_grads = torch.sort(stacked_grads, dim=0).values
        trimmed_grads = sorted_grads[1:-1]
        # 计算剩余值的均值
        mean_grads = torch.mean(trimmed_grads, dim=0)
        # 存储均值梯度
        median_grads_dict[grad_key] = mean_grads
    return median_grads_dict


def filter_and_select(result, midden_client_agg):
    selected_keys = [key for key, value in result.items() if value > 0.9]

    selected_midden_client_agg = {key: midden_client_agg[key] for key in selected_keys if key in midden_client_agg}

    return selected_midden_client_agg


def getAverageAggWellGlobalModel(weights_dicts_origin):
    weights_dicts = copy.deepcopy(weights_dicts_origin)
    # 确定客户端数量
    num_clients = len(weights_dicts)

    # 创建一个空字典，用于存储平均权重参数
    averaged_weights_dict = {}

    # 遍历每个客户端的权重参数
    for k, client_weights_dict in weights_dicts.items():
        for key, value in client_weights_dict.items():
            if key not in averaged_weights_dict:
                # 如果键不存在于平均权重字典中，则将其添加并初始化为当前客户端的权重参数
                averaged_weights_dict[key] = value.clone().detach().float()  # 使用clone().detach()创建张量
            else:
                # 如果键已存在于平均权重字典中，则将当前客户端的权重参数加到已存在的值上
                averaged_weights_dict[key] = averaged_weights_dict[key] + value.clone().detach().float()  # 使用clone().detach()创建张量

    # 将每个位置的值除以客户端数量，得到平均值
    for key in averaged_weights_dict:
        averaged_weights_dict[key] = averaged_weights_dict[key] / num_clients
        averaged_weights_dict[key] = averaged_weights_dict[key].float()

    return averaged_weights_dict


def change_label(tensor):
    # 计算需要变换符号的数据数量
    # 计算需要变换符号的数据数量
    num_to_change = int(0.3 * tensor.numel())

    # 生成随机布尔张量，用于选择需要变换符号的元素
    mask = torch.rand(tensor.size()) < num_to_change / tensor.numel()

    # 将需要变换符号的元素乘以-1
    tensor[mask] *= -1

    return tensor


def untarget_attack(tensor, ratio=0.2):
    # 计算需要变换符号的数据数量
    num_to_change = int(tensor.numel() * ratio)

    # 生成需要变换符号的索引
    indices = torch.randperm(tensor.numel())[:num_to_change]

    # 生成随机浮点数张量，范围在-0.1到0.2之间
    random_values = torch.rand(num_to_change) * 0.3 - 2

    tensor = tensor.float()  # 将整个张量转换为浮点数类型
    tensor.view(-1)[indices] = random_values

    return tensor


def poison_gradients(grads_dict, title):
    if title == 'sign-attack':
        for key, tensor in grads_dict.items():
            # 对每个键值对的值进行部分翻转毒化
            grads_dict[key] = change_label(tensor)
    else:
        for key, tensor in grads_dict.items():
            # 对每个键值对的值进行部分翻转毒化
            grads_dict[key] = untarget_attack(tensor)

    return grads_dict


def save_images(predictions, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    class_images = [[] for _ in range(10)]

    for data, predicted, label in predictions:
        for i in range(len(data)):
            if len(class_images[predicted[i]]) < 10:
                class_images[predicted[i]].append((data[i], predicted[i], label[i]))

    for class_idx, images_idx in enumerate(class_images):
        class_dir = os.path.join(output_dir, f'class_{class_idx}')
        os.makedirs(class_dir, exist_ok=True)
        for i, (image, pred, true_label) in enumerate(images_idx):
            image_path = os.path.join(class_dir, f'image_{i + 1}.png')
            image_np = image.squeeze().detach().cpu().numpy()
            image_np = image_np.reshape(28, 28)

            plt.imshow(image_np, cmap='gray')
            plt.axis('off')
            plt.savefig(image_path)
            plt.close()
            # print(f'Saved image {i + 1} to {image_path}')
    # plt.close()


if __name__ == "__main__":
    # Create a sample dictionary
    # all_poison_accurate = {}
    #
    # # Populate the dictionary with sample values
    # for i in range(5):
    #     poison_percent = i * 0.1
    #     record_accurate_poison = {}
    #     for j in range(3):
    #         tensor_array = torch.tensor([i, j])
    #         record_accurate_poison[j] = tensor_array
    #     all_poison_accurate[poison_percent] = record_accurate_poison
    #
    # # Print the resulting dictionary
    # print(all_poison_accurate)
    #
    # # Create a DataFrame to store the data
    # df = pd.DataFrame(index=range(5), columns=range(3))
    #
    # for poison_percent, record_accurate_poison in all_poison_accurate.items():
    #     for i, tensor_array in record_accurate_poison.items():
    #         # 将浮点数索引转换为整数索引
    #         poison_percent_index = int(poison_percent * 10)
    #         i_index = int(i)
    #
    #         # 将tensor的值作为数据
    #         df.iloc[poison_percent_index, i_index] = tensor_array.cpu().numpy()

    # Print the resulting DataFrame
    tensor = torch.tensor([1, 2, 3, 4, 5])
    flipped_labels = flip_labels(tensor)
    print(flipped_labels)
