import sys
import os
import pandas as pd
import numpy as np
import torch

from classifier_hybrid.net.classifier import Classifier
from compute_aff_features.compute_features import compute_features
from compute_aff_features.normalize_features import normalize_features




def load_hybrid_model(
    model_dir,
    in_channels=3,
    in_features=48,  # Số lượng affective features, có thể cần điều chỉnh
    num_classes=4,
    joints=16,
    temporal_kernel_size=75,
    device='cuda:0'
):
    # Khởi tạo graph_dict
    # graph_dict = {'strategy': 'spatial', 'layout': 'ntu-rgb+d', 'num_node': joints}
    graph_dict = {'strategy': 'spatial'}

    # Khởi tạo model
    model = Classifier(
        in_channels=in_channels,
        in_features=in_features,
        num_classes=num_classes,
        graph_args=graph_dict,
        temporal_kernel_size=temporal_kernel_size
    )
    model = model.to(device)

    # Tìm file trọng số tốt nhất
    weight_files = [f for f in os.listdir(model_dir) if f.endswith('.pth.tar')]
    assert weight_files, "Không tìm thấy file trọng số"
    # Chọn file acc cao nhất
    best_weight = sorted(weight_files, key=lambda x: float(x.split('_acc')[1].split('_')[0]), reverse=True)[0]
    weight_path = os.path.join(model_dir, best_weight)

    # Load trọng số
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model



def predict_emotion_from_gait(model, gait, device='cuda:0'):
    """
    gait: numpy array shape (T, joints*coords) hoặc (T, joints, coords)
    """
    in_channels = 3
    joints = 16

    aff = []
    aff_norma = []
    time_step = 1.0 / 30.0
    aff.append(compute_features(gait,time_step))
    normalize_features(aff, aff_norma) #


    aff = aff_norma

    # Lấy trung bình theo thời gian (T) để ra vector aff (48,)
    aff_vec = aff.mean(axis=0)

    # Chuẩn hóa shape cho model hybrid
    T = gait.shape[0]
    gait_for_model = gait.transpose(2,0,1)  # (coords, T, joints)
    gait_for_model = gait_for_model[np.newaxis, ...]  # (1, coords, T, joints)
    gait_for_model = gait_for_model[..., np.newaxis]  # (1, coords, T, joints, 1)
    gait_tensor = torch.from_numpy(gait_for_model).float().to(device)
    aff_tensor = torch.from_numpy(aff_vec).float().unsqueeze(0).to(device)  # (1, 48)

    with torch.no_grad():
        output = model(aff_tensor, gait_tensor)
        pred_label = torch.argmax(output, dim=1).item()
    return pred_label

