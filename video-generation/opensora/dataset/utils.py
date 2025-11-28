import cv2
import numpy as np
import matplotlib
import math
import torch
import random
import copy
from dwpose_utils.dwpose_detector import build_pose_dict
from torch.nn import functional as F
eps = 0.01

DROP_FACE_POINTS = {0, 14, 15, 16, 17}
DROP_UPPER_POINTS = {0, 14, 15, 16, 17, 2, 1, 5, 3, 6}
DROP_LOWER_POINTS = {8, 9, 10, 11, 12, 13}

def generate_pose_plan(pose_aug_probs):
    p1, p2, p3, p4 = pose_aug_probs
    r = random.random()
    if r < p1:
        return {"mode": "drop_point", "point_idx": random.randint(0, 17)}
    elif r < p1 + p2:
        return {"mode": "stretch_limb", "limb_idx": random.randint(0, 18),
                "stretch_scale": random.uniform(0.5, 2.0)}
    elif r < p1 + p2 + p3:
        region_mode = random.choice(["face", "upper", "lower"])
        if region_mode == "face":
            pts = DROP_FACE_POINTS
        elif region_mode == "upper":
            pts = DROP_UPPER_POINTS
        else:
            pts = DROP_LOWER_POINTS
        return {"mode": "drop_region", "region": region_mode, "points": pts}
    else:
        return {"mode": "none"}

def get_stickwidth(W, H, stickwidth=4):
    if max(W, H) < 512:
        ratio = 1.0
    elif max(W, H) < 1080:
        ratio = 1.5
    elif max(W, H) < 2160:   
        ratio = 2.0
    elif max(W, H) < 3240:   
        ratio = 2.5
    elif max(W, H) < 4320:  
        ratio = 3.5
    elif max(W, H) < 5400:   
        ratio = 4.5
    else:                 
        ratio = 4.0
    return int(stickwidth * ratio)


def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]


def draw_bodypose_aligned(canvas, candidate, subset, score, 
                          plan=None,   # 新增，用于全视频固定增强
                          random_drop=False):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)
    stickwidth = get_stickwidth(W, H, stickwidth=3)

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
        [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
        [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18]]
    colors = [
        [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],
        [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255],
        [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
        [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    HIDE_JOINTS = set()
    stretch_limb_idx = None
    stretch_scale = None
    if plan:
        if plan["mode"] == "drop_point":
            HIDE_JOINTS.add(plan["point_idx"])
        elif plan["mode"] == "drop_region":
            HIDE_JOINTS |= set(plan["points"])
        elif plan["mode"] == "stretch_limb":
            stretch_limb_idx = plan["limb_idx"]
            stretch_scale = plan["stretch_scale"]

    hide_joint = np.zeros_like(subset, dtype=bool)

    for i in range(17):
        for n in range(len(subset)):
            idx_pair = limbSeq[i]

            if any(j in HIDE_JOINTS for j in idx_pair):
                continue

            index = subset[n][np.array(idx_pair) - 1]
            conf = score[n][np.array(idx_pair) - 1]
            if -1 in index:
                continue
            # color lighten
            alpha = max(conf[0] * conf[1], 0) if conf[0]>0 and conf[1]>0 else 0.35

            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)

            if stretch_limb_idx == i:
                vec_x = X[1] - X[0]
                vec_y = Y[1] - Y[0]
                X[1] = X[0] + vec_x * stretch_scale
                Y[1] = Y[0] + vec_y * stretch_scale
                hide_joint[n, idx_pair[1]-1] = True

            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0]-X[1])**2 + (Y[0]-Y[1])**2) ** 0.5
            angle = math.degrees(math.atan2(X[0]-X[1], Y[0]-Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)),
                                       (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], alpha))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        if i in HIDE_JOINTS:
            continue
        for n in range(len(subset)):
            if hide_joint[n, i]:
                continue
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            # head_change=True
            alpha = 0 if conf==-2 else max(conf, 0)
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (x, y), stickwidth,
                       alpha_blend_color(colors[i], alpha), thickness=-1)

    return canvas


def draw_handpose_aligned(canvas, all_hand_peaks, all_hand_scores, draw_th=0.3,):
    H, W, C = canvas.shape
    stickwidth = get_stickwidth(W, H, stickwidth=2)
    line_thickness = get_stickwidth(W, H, stickwidth=2)

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):
        for ie, e in enumerate(edges):
            if scores[e[0]] < draw_th or scores[e[1]] < draw_th:
                    continue
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)

            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                # 原来是2
                cv2.line(canvas, (x1, y1), (x2, y2),
                         matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score, thickness=line_thickness)

        for i, keyponit in enumerate(peaks):
            if scores[i] < draw_th:
                continue

            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            score = int(scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), stickwidth, (0, 0, score), thickness=-1)
    return canvas


def draw_facepose_aligned(canvas, all_lmks, all_scores, draw_th=0.3,face_change=False):
    H, W, C = canvas.shape
    stickwidth = get_stickwidth(W, H, stickwidth=2)
    SKIP_IDX = set(range(0, 17))
    SKIP_IDX |= set(range(27, 36))    
    
    for lmks, scores in zip(all_lmks, all_scores):
        for idx, (lmk, score) in enumerate(zip(lmks, scores)):
            # 跳过下巴点
            if idx in SKIP_IDX:
                continue
            if score < draw_th:
                continue
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            conf = int(score * 255)
            # color lighten
            if face_change:
                conf = int(conf * 0.35)

            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), stickwidth, (conf, conf, conf), thickness=-1)
    return canvas


def draw_pose_aligned(pose, H, W, ref_w=2160,
                      without_face=False,
                      pose_plan=None,
                      head_change=False,
                      face_change=False):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    body_score = bodies['score'].copy()
    # control color
    if head_change:
        target_joints = [0, 14, 15, 16, 17]
        body_score[:, target_joints] = -2


    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    canvas = draw_bodypose_aligned(canvas, candidate, subset,
                                   score=body_score,
                                   plan=pose_plan,)

    # drop上半身 → 手不要画
    if pose_plan and pose_plan["mode"] == "drop_region" and pose_plan.get("region") == "upper":
        pass
    else:
        canvas = draw_handpose_aligned(canvas, hands, pose['hands_score'])

    # drop脸 → 不画脸
    if pose_plan and pose_plan["mode"] == "drop_region" and pose_plan.get("region") in {"upper","face"}:
        pass
    elif not without_face:
        canvas = draw_facepose_aligned(canvas, faces, pose['faces_score'],face_change=face_change)
    # print(W,H)
    return cv2.resize(canvas, (W, H))

# 训练中并没有用到
def generate_pose_plan(pose_aug_probs):
    p1, p2, p3, p4 = pose_aug_probs
    r = random.random()
    if r < p1:
        return {"mode": "drop_point", "point_idx": random.randint(0, 17)}
    elif r < p1 + p2:
        return {"mode": "stretch_limb", "limb_idx": random.randint(0, 18),
                "stretch_scale": random.uniform(0.5, 2.0)}
    elif r < p1 + p2 + p3:
        region_mode = random.choice(["face", "upper", "lower"])
        if region_mode == "face":
            pts = DROP_FACE_POINTS
        elif region_mode == "upper":
            pts = DROP_UPPER_POINTS
        else:
            pts = DROP_LOWER_POINTS
        return {"mode": "drop_region", "region": region_mode, "points": pts}
    else:
        return {"mode": "none"}

def get_batch_pose(pose_data, train_indices, H, W, 
                   without_face=False, 
                   pose_aug_flag=False,
                   head_change=False,
                   face_change=False):

    if pose_aug_flag:
        pose_plan = generate_pose_plan((0.1, 0.1, 0.3, 0.5))
    else:
        pose_plan = None

    pose_batch = []
    for i in train_indices:
        frame_pose_img = draw_pose_aligned(pose_data[i], H, W, without_face=without_face,pose_plan=pose_plan,head_change=head_change,face_change=face_change)
        pose_batch.append(frame_pose_img)
    return np.array(pose_batch)



def select_refimg_index(refimg_indices, train_indices):
    valid_indices = [idx for idx in refimg_indices if idx < train_indices[0] or idx > train_indices[-1]]
    if not valid_indices:
        valid_indices = refimg_indices

    return np.random.choice(valid_indices)


def draw_face_mask(all_face_bbox, train_indices, real_t, ori_height, ori_width, for_vae = True):
    vae_t  = 1 + (real_t-1) // 4
    if len(all_face_bbox) == 0:
        if for_vae:
            return torch.ones((vae_t, ori_height, ori_width), dtype=torch.float32)
        else:
            return torch.ones((real_t, ori_height, ori_width), dtype=torch.float32)
    # 如果存在facebbox
    if for_vae:
        mask = torch.zeros((vae_t, ori_height, ori_width), dtype=torch.float32)
    else:
        mask = torch.zeros((real_t, ori_height, ori_width), dtype=torch.float32)

    for i, idx in enumerate(train_indices):
        bbox = all_face_bbox[idx]  # 获取对应的 bbox
        if len(bbox) == 0:
            continue
        x1, y1, x2, y2 = map(int,bbox)

        if for_vae:
            compressed_idx = (i + 3) // 4  # 每 4 帧压缩成一帧
            # 并集
            mask[compressed_idx, y1:y2, x1:x2] = 1
        else:
            mask[i, y1:y2, x1:x2] = 1
    return mask


def draw_face_mask_frompose(pose_data, train_indices, real_t,
                   ori_height, ori_width,
                   for_vae=True,           # 是否 4→1 压缩
                   score_thresh=0.8,       # 关键点平均置信度阈值
                   expand_ratio=0.1):      # 人脸 box 适当放大一点

    vae_t = 1 + (real_t - 1) // 4
    if for_vae:
        mask = torch.zeros((vae_t, ori_height, ori_width), dtype=torch.float32)
        valid_flags = [True] * vae_t
    else:
        mask = torch.zeros((real_t, ori_height, ori_width), dtype=torch.float32)

    for i, idx in enumerate(train_indices):
        pose = pose_data[idx]
        faces  = pose.get('faces',  pose.get('face',  []))
        scores = pose.get('faces_score', pose.get('face_score', []))

        if len(scores) == 0 or len(faces) == 0:
            if for_vae:
                valid_flags[(i + 3) // 4] = False
            continue

        frame_valid = True
        frame_mask = torch.zeros((ori_height, ori_width), dtype=torch.float32)

        for peaks, sc in zip(faces, scores):
            if np.mean(sc) < score_thresh:
                frame_valid = False
                break
            xys = np.array(peaks)             
            if xys.size == 0:
                continue
            x1, y1 = int(np.min(xys[:, 0] * ori_width)),  int(np.min(xys[:, 1] * ori_height))
            x2, y2 = int(np.max(xys[:, 0] * ori_width)),  int(np.max(xys[:, 1] * ori_height))

            # 额外抬高顶部
            extra_top = int((y2 - y1) * 0.4)
            y1 = max(0, y1 - extra_top)

            x1, y1, x2, y2 = expand_bbox(
                x1, y1, x2, y2,
                big=expand_ratio, small=expand_ratio,
                frame_width=ori_width, frame_height=ori_height
            )

            frame_mask[y1:y2, x1:x2] = 1

        if not frame_valid:
            if for_vae:
                valid_flags[(i + 3) // 4] = False
            continue

        if for_vae:
            compressed_idx = (i + 3) // 4
            mask[compressed_idx] = torch.maximum(mask[compressed_idx], frame_mask)
        else:
            mask[i] = frame_mask

    if for_vae:
        for t, valid in enumerate(valid_flags):
            if not valid:
                mask[t] = 0

    return mask
    

def draw_hand_mask(pose_data, train_indices, real_t, ori_height, ori_width, for_vae=True, score_thresh=0.8, expand_ratio=0.2):
    vae_t = 1 + (real_t - 1) // 4
    if for_vae:
        mask = torch.zeros((vae_t, ori_height, ori_width), dtype=torch.float32)
        valid_flags = [True] * vae_t
    else:
        mask = torch.zeros((real_t, ori_height, ori_width), dtype=torch.float32)

    for i, idx in enumerate(train_indices):
        pose = pose_data[idx]
        hands = pose.get('hands', [])
        hand_scores = pose.get('hands_score', [])
        if len(hand_scores) == 0 or len(hands) == 0:
            if for_vae:
                valid_flags[(i + 3) // 4] = False
            continue

        frame_valid = True
        # 用一个临时mask存每只手的mask，最后叠加
        frame_mask = torch.zeros((ori_height, ori_width), dtype=torch.float32)
        for peaks, scores in zip(hands, hand_scores):
            if np.mean(scores) < score_thresh:
                frame_valid = False
                break
            xys = np.array(peaks)
            if xys.size == 0:
                continue
            x1, y1 = int(np.min(xys[:, 0] * ori_width)), int(np.min(xys[:, 1] * ori_height))
            x2, y2 = int(np.max(xys[:, 0] * ori_width)), int(np.max(xys[:, 1] * ori_height))

            x1, y1, x2, y2 = expand_bbox(
                x1, y1, x2, y2,
                big=expand_ratio, small=expand_ratio,
                frame_width=ori_width,
                frame_height=ori_height
            )

            frame_mask[y1:y2, x1:x2] = 1  # 给每只手单独mask
        if not frame_valid:
            if for_vae:
                valid_flags[(i + 3) // 4] = False
            continue
        # 合并frame_mask到大mask
        if for_vae:
            compressed_idx = (i + 3) // 4
            mask[compressed_idx] = torch.maximum(mask[compressed_idx], frame_mask)
        else:
            mask[i] = frame_mask

    if for_vae:
        for t, valid in enumerate(valid_flags):
            if not valid:
                mask[t] = 0  # 四帧内只要有一帧不合格，就清空对应压缩帧mask
    return mask

def expand_bbox(x1, y1, x2, y2, big=0.1, small=0.4,max_expand_pixels=None, frame_width=None, frame_height=None):
    width = x2 - x1
    height = y2 - y1
    if width > height:
        width_expand_ratio = big
        height_expand_ratio = small
    else:
        width_expand_ratio = small
        height_expand_ratio = big
    
    expand_width = width * width_expand_ratio
    expand_height = height * height_expand_ratio
    
    if max_expand_pixels is not None:
        expand_width = min(expand_width, max_expand_pixels)
        expand_height = min(expand_height, max_expand_pixels)
    
    x1_new = max(0, x1 - expand_width)
    y1_new = max(0, y1 - expand_height)
    x2_new = x2 + expand_width
    y2_new = y2 + expand_height
    
    # 如果提供了图像宽度和高度，限制扩展后的坐标
    if frame_width is not None:
        x2_new = min(x2_new, frame_width)
    if frame_height is not None:
        y2_new = min(y2_new, frame_height)
    
    return int(x1_new), int(y1_new), int(x2_new), int(y2_new)


#ocr_bbox可能有多个, ocr_bbox需要expand,目标:ocr部分为0
def draw_ocr_mask(all_ocr_bbox, train_indices, real_t, ori_height, ori_width, for_vae = True):
    vae_t  = 1 + (real_t-1) // 4
    if len(all_ocr_bbox) == 0:
        if for_vae:
            return torch.ones((vae_t, ori_height, ori_width), dtype=torch.float32)
        else:
            return torch.ones((real_t, ori_height, ori_width), dtype=torch.float32)
    # 如果存在ocrbbox
    if for_vae:
        mask = torch.ones((vae_t, ori_height, ori_width), dtype=torch.float32)
    else:
        mask = torch.ones((real_t, ori_height, ori_width), dtype=torch.float32)

    for i, idx in enumerate(train_indices):
        bbox_list = all_ocr_bbox[idx]
        if len(bbox_list) == 0:
            continue
        for bbox in bbox_list:  # 遍历所有的 bbox
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1, x2, y2 = expand_bbox(
                x1, y1, x2, y2,
                max_expand_pixels=20,
                frame_width=ori_width,
                frame_height=ori_height
            )
            if for_vae:
                compressed_idx = (i + 3) // 4  # 每 4 帧压缩成一帧
                # 并集
                mask[compressed_idx, y1:y2, x1:x2] = 0
            else:
                mask[i, y1:y2, x1:x2] = 0
    return mask

def crop_tensors(tensors, bbox):
    if not bbox:
        return tensors
    x1, y1, x2, y2 = bbox
    return [t[..., y1:y2, x1:x2] for t in tensors]

def get_bbox_from_mask(mask: torch.Tensor):
    if not mask.any():
        return None
    combined_mask = mask.any(dim=0).squeeze(0)
    coords = combined_mask.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    y_min, x_min = coords.min(dim=0).values
    y_max, x_max = coords.max(dim=0).values
    return int(x_min), int(y_min), int(x_max), int(y_max)




def apply_random_cropping_strategy(
    video, pose, reference_frame, face_mask, ocr_mask, hand_mask, 
    reference_pose_tensor,
    vid_face_bbox, ref_face_bbox,
    real_h, real_w,
    pose_cfg=0.2, refimg_cfg=0.0,refimg_crop=0.2, video_crop=0.2
):
    p = random.random()

    if p < pose_cfg:
        # 场景A: CFG - 置零 pose 和 reference_frame
        pose = torch.zeros_like(pose)
        reference_frame = torch.zeros_like(reference_frame)
        reference_pose_tensor = torch.zeros_like(reference_pose_tensor)

    elif p < pose_cfg+refimg_cfg:
        # 场景B: CFG - 置零  reference_frame
        reference_frame = torch.zeros_like(reference_frame)
        reference_pose_tensor = torch.zeros_like(reference_pose_tensor)

    elif p < pose_cfg + refimg_cfg + video_crop and vid_face_bbox is not None:
        f_x1, f_y1, f_x2, f_y2 = vid_face_bbox
        H, W = video.shape[-2:]
        
        y1_crop = max(0, f_y1 - int((f_y2 - f_y1) * 0.1))
        y2_crop = random.randint(int(H * 0.7), H)

        if y2_crop > y1_crop + 10: # 保证有效高度
            crop_h = y2_crop - y1_crop
            crop_w = int(crop_h * (real_w / real_h))

            x_start_range = max(0, f_x2 - crop_w)
            x_end_range = min(W - crop_w, f_x1)
            
            if x_start_range < x_end_range:
                x1_crop = random.randint(x_start_range, x_end_range)
                x2_crop = x1_crop + crop_w
                final_bbox = (x1_crop, y1_crop, x2_crop, y2_crop)
                video, pose, face_mask, ocr_mask, hand_mask = crop_tensors(
                    [video, pose, face_mask, ocr_mask, hand_mask], final_bbox
                )

    elif p < pose_cfg + refimg_cfg + video_crop + refimg_crop and ref_face_bbox is not None:
        f_x1, f_y1, f_x2, f_y2 = ref_face_bbox
        H, W = reference_frame.shape[-2:]

        y1_crop = max(0, f_y1 - int((f_y2 - f_y1) * 0.1))
        y2_crop_min = f_y2 + int((f_y2 - f_y1) * 0.3)
        y2_crop = random.randint(min(y2_crop_min, H - 1), H - 1)

        if y2_crop > y1_crop + 10:
            crop_h = y2_crop - y1_crop
            crop_w = int(crop_h * (real_w / real_h))
            x_start_range = max(0, f_x2 - crop_w)
            x_end_range = min(W - crop_w, f_x1)
            if x_start_range < x_end_range:
                x1_crop = random.randint(x_start_range, x_end_range)
                x2_crop = x1_crop + crop_w
                final_bbox = (x1_crop, y1_crop, x2_crop, y2_crop)
                reference_frame, reference_pose_tensor = crop_tensors([reference_frame, reference_pose_tensor], final_bbox)

    return video, pose, reference_frame, face_mask, ocr_mask, hand_mask, reference_pose_tensor 


def generate_random_rectangle_mask(H, W, max_rects=1, max_size_ratio=0.2):
    mask = torch.zeros((1, 1, H, W), dtype=torch.uint8)
    for _ in range(random.randint(1, max_rects)):
        rect_h = random.randint(int(H * 0.05), int(H * max_size_ratio))
        rect_w = random.randint(int(W * 0.05), int(W * max_size_ratio))
        
        start_y = random.randint(0, H - rect_h)
        start_x = random.randint(0, W - rect_w)
        
        mask[:, :, start_y : start_y + rect_h, start_x : start_x + rect_w] = 1
            
    return mask

# use mask
def apply_random_cropping_strategy_v2(
    video, pose, reference_frame, face_mask, ocr_mask, hand_mask, 
    reference_pose_tensor,
    vid_face_bbox, ref_face_bbox,
    real_h, real_w,
    pose_cfg=0.0, refimg_cfg=0.0, refimg_crop=0.0, random_zoomin=0.0,ref_keep=0.0,
):
    reference_frame_ori  =  reference_frame.clone()
    _, _, H, W = reference_frame.shape
    inpaint_mask = torch.zeros((1, 1, H, W),dtype=torch.uint8)
    
    p = random.random()

    if p < pose_cfg:
        pose = torch.zeros_like(pose)
        reference_frame = torch.zeros_like(reference_frame)
        reference_pose_tensor = torch.zeros_like(reference_pose_tensor)
        return video, pose, reference_frame, face_mask, ocr_mask, hand_mask, reference_pose_tensor, inpaint_mask

    elif p < pose_cfg + refimg_cfg:
        reference_frame = torch.zeros_like(reference_frame)
        # reference_pose_tensor = torch.zeros_like(reference_pose_tensor)
        return video, pose, reference_frame, face_mask, ocr_mask, hand_mask, reference_pose_tensor, inpaint_mask

    elif p < pose_cfg + refimg_cfg + refimg_crop and ref_face_bbox is not None:
        ratio_bottom = 0.9
        f_x1, f_y1, f_x2, f_y2 = ref_face_bbox
        H, W = reference_frame.shape[-2:]

        y1_crop = max(0, f_y1 - int((f_y2 - f_y1) * 0.1))
        y2_crop_min = f_y2 + int((f_y2 - f_y1) * 0.1)
        y2_crop_max = int(H * ratio_bottom)
        if y2_crop_max < y2_crop_min: # 极端情况下保证有合法区间
            y2_crop_max = y2_crop_min

        y2_crop = random.randint(y2_crop_min, y2_crop_max)
        # y2_crop = random.randint(min(y2_crop_min, H - 1), H - 1)

        if y2_crop > y1_crop + 10:
            crop_h = y2_crop - y1_crop
            crop_w = int(crop_h * (real_w / real_h))
            x_start_range = max(0, f_x2 - crop_w)
            x_end_range = min(W - crop_w, f_x1)
            if x_start_range < x_end_range:
                x1_crop = random.randint(x_start_range, x_end_range)
                x2_crop = x1_crop + crop_w
                final_bbox = (x1_crop, y1_crop, x2_crop, y2_crop)
                # x1, y1, x2, y2 = final_bbox
                mask_keep = torch.zeros((1, 1, H, W),dtype=torch.uint8)
                mask_keep[..., y1_crop:y2_crop, x1_crop:x2_crop] = 1

                # 更新 inpaint_mask (bbox外为1，内为0)
                inpaint_mask = 1 - mask_keep
                # mask ref and ref pose
                reference_frame = reference_frame * mask_keep
                reference_pose_tensor = reference_pose_tensor * mask_keep
    else:
        if random.random() < 0.2:
            random_mask_to_add = generate_random_rectangle_mask(H, W).to(inpaint_mask.device)
            inpaint_mask = random_mask_to_add
            mask_keep = (1 - inpaint_mask).to(reference_frame.dtype)
            reference_frame = reference_frame * mask_keep
            reference_pose_tensor = reference_pose_tensor * mask_keep

    if random.random() < ref_keep:
        pose = torch.zeros_like(pose)
        reference_pose_tensor = torch.zeros_like(reference_pose_tensor)
        if video.shape[0] == 1: # T=1
            video = reference_frame_ori

        return video, pose, reference_frame, face_mask, ocr_mask, hand_mask, reference_pose_tensor, inpaint_mask
    
    if random_zoomin > 0.0:
        q = random.random()
        zoom_scale = 1.0 + random.uniform(0.0, 0.2)

        def zoom_in_center_crop(x, scale):
            Hx, Wx = x.shape[-2], x.shape[-1]
            newH = max(1, int(round(Hx * scale)))
            newW = max(1, int(round(Wx * scale)))
            if newH == Hx and newW == Wx:
                return x
            x_up = F.interpolate(x, size=(newH, newW), mode='nearest')
            top = max(0, (newH - Hx) // 2)
            left = max(0, (newW - Wx) // 2)
            return x_up[..., top:top + Hx, left:left + Wx]

        if q < random_zoomin * 0.5:
            reference_frame = zoom_in_center_crop(reference_frame, zoom_scale)
            reference_pose_tensor = zoom_in_center_crop(reference_pose_tensor, zoom_scale)
            inpaint_mask = zoom_in_center_crop(inpaint_mask.float(), zoom_scale).to(inpaint_mask.dtype)

        elif q < random_zoomin:
            video = zoom_in_center_crop(video, zoom_scale)
            pose = zoom_in_center_crop(pose, zoom_scale)
            face_mask = zoom_in_center_crop(face_mask.float(), zoom_scale).to(face_mask.dtype)
            ocr_mask = zoom_in_center_crop(ocr_mask.float(), zoom_scale).to(ocr_mask.dtype)
            hand_mask = zoom_in_center_crop(hand_mask.float(), zoom_scale).to(hand_mask.dtype)

    return video, pose, reference_frame, face_mask, ocr_mask, hand_mask, reference_pose_tensor, inpaint_mask





def _calculate_center_crop_bbox(H, W, zoom_factor=0.7):
    crop_h = int(H * zoom_factor)
    crop_w = int(W * zoom_factor)
    y1 = (H - crop_h) // 2
    x1 = (W - crop_w) // 2
    return (x1, y1, x1 + crop_w, y1 + crop_h)

def _calculate_smart_crop_bbox(bbox, H, W, real_h, real_w):
    ratio_bottom = 0.75
    f_x1, f_y1, f_x2, f_y2 = bbox
    # 使用你之前为 refimg 设计的随机裁剪逻辑
    y1_crop = max(0, f_y1 - int((f_y2 - f_y1) * 0.1))
    y2_crop_min = f_y2 + int((f_y2 - f_y1) * 0.1)
    y2_crop_max = int(H * ratio_bottom)
    if y2_crop_max < y2_crop_min: # 极端情况下保证有合法区间
        y2_crop_max = y2_crop_min
    y2_crop = random.randint(y2_crop_min, y2_crop_max)
    # y2_crop = random.randint(min(y2_crop_min, H - 1), H - 1)

    if y2_crop <= y1_crop + 10:
        return None

    crop_h = y2_crop - y1_crop
    crop_w = int(crop_h * (real_w / real_h))
    x_start_range = max(0, f_x2 - crop_w)
    x_end_range = min(W - crop_w, f_x1)
    
    if x_start_range >= x_end_range:
        return None
        
    x1_crop = random.randint(x_start_range, x_end_range)
    x2_crop = x1_crop + crop_w
    return (x1_crop, y1_crop, x2_crop, y2_crop)

# mask
def apply_image_augmentation_strategy_v2(
    image, pose, reference_frame, reference_pose,
    tgt_face_mask,
    is_same_image,
    real_h, real_w,
    pose_cfg=0.0,
    refimg_cfg = 0.0,
    refimg_crop = 0.0,
    random_zoomin=0.0,
    ref_keep=0.0,
):

    reference_frame_ori = reference_frame.clone()
    _, _, H, W = reference_frame.shape
    inpaint_mask = torch.zeros((1, 1, H, W),dtype=torch.uint8)
    
    if random.random() < pose_cfg:
        pose = torch.zeros_like(pose)
        reference_frame = torch.zeros_like(reference_frame)
        reference_pose = torch.zeros_like(reference_pose)

    elif random.random() < pose_cfg + refimg_cfg:
        reference_frame = torch.zeros_like(reference_frame)
        # reference_pose = torch.zeros_like(reference_pose)

    # diff image
    elif refimg_crop == 0.0 or not is_same_image:
        return image, pose, reference_frame, reference_pose, tgt_face_mask, inpaint_mask
    
    else:
        bbox = get_bbox_from_mask(tgt_face_mask)
        final_bbox = None
        if bbox:
            final_bbox = _calculate_smart_crop_bbox(bbox, reference_frame.shape[-2], reference_frame.shape[-1], real_h, real_w)
        if final_bbox:
            x1_crop, y1_crop, x2_crop, y2_crop = final_bbox
            mask_keep = torch.zeros((1, 1, H, W),dtype=torch.uint8)
            mask_keep[...,  y1_crop:y2_crop, x1_crop:x2_crop] = 1
            inpaint_mask = 1 - mask_keep
            reference_frame = reference_frame * mask_keep
            reference_pose = reference_pose * mask_keep

            
        else:
            H, W = reference_frame.shape[-2:]
            center_crop_bbox = _calculate_center_crop_bbox(H, W)
            x1_crop, y1_crop, x2_crop, y2_crop = center_crop_bbox
            mask_keep = torch.zeros((1, 1, H, W),dtype=torch.uint8)
            mask_keep[..., y1_crop:y2_crop, x1_crop:x2_crop] = 1
            inpaint_mask = 1 - mask_keep
            reference_frame = reference_frame * mask_keep
            reference_pose = reference_pose * mask_keep

    if random.random() < ref_keep:
        pose = torch.zeros_like(pose)
        reference_pose = torch.zeros_like(reference_pose)
        image = reference_frame_ori
        return image, pose, reference_frame, reference_pose, tgt_face_mask, inpaint_mask
    if random_zoomin > 0.0:
        q = random.random()
        zoom_scale = 1.0 + random.uniform(0.0, 0.3)

        def zoom_in_center_crop(x, scale):
            Hx, Wx = x.shape[-2], x.shape[-1]
            newH = max(1, int(round(Hx * scale)))
            newW = max(1, int(round(Wx * scale)))
            if newH == Hx and newW == Wx:
                return x
            x_up = F.interpolate(x, size=(newH, newW), mode='nearest')
            top = max(0, (newH - Hx) // 2)
            left = max(0, (newW - Wx) // 2)
            return x_up[..., top:top + Hx, left:left + Wx]

        if q < random_zoomin * 0.5:
            reference_frame = zoom_in_center_crop(reference_frame, zoom_scale)
            reference_frame = zoom_in_center_crop(reference_frame, zoom_scale)
            inpaint_mask = zoom_in_center_crop(inpaint_mask.float(), zoom_scale).to(inpaint_mask.dtype)

        elif q < random_zoomin:
            image = zoom_in_center_crop(image, zoom_scale)
            pose = zoom_in_center_crop(pose, zoom_scale)
            tgt_face_mask = zoom_in_center_crop(tgt_face_mask.float(), zoom_scale).to(tgt_face_mask.dtype)
    return image, pose, reference_frame, reference_pose, tgt_face_mask, inpaint_mask


def apply_image_augmentation_strategy(
    image, pose, reference_frame, reference_pose,
    tgt_face_mask,
    is_same_image,
    real_h, real_w,
    pose_cfg=0.2,
    refimg_cfg = 0.0,
    refimg_crop = 0.0
):
    if random.random() < pose_cfg:
        pose = torch.zeros_like(pose)
        reference_frame = torch.zeros_like(reference_frame)
        reference_pose = torch.zeros_like(reference_pose)
        return image, pose, reference_frame, reference_pose, tgt_face_mask
    
    if random.random() < pose_cfg + refimg_cfg:
        reference_frame = torch.zeros_like(reference_frame)
        reference_pose = torch.zeros_like(reference_pose)
        return image, pose, reference_frame, reference_pose, tgt_face_mask
    
    if refimg_crop == 0.0 or not is_same_image:
        return image, pose, reference_frame, reference_pose, tgt_face_mask

    bbox = get_bbox_from_mask(tgt_face_mask)
    final_bbox = None
    if bbox:
        final_bbox = _calculate_smart_crop_bbox(bbox, reference_frame.shape[-2], reference_frame.shape[-1], real_h, real_w)
    if final_bbox:
        if random.random() < 0.5:
            reference_frame, reference_pose = crop_tensors([reference_frame, reference_pose], final_bbox)
        else:
            image, pose, tgt_face_mask = crop_tensors([image, pose, tgt_face_mask], final_bbox)
    else:
        H, W = reference_frame.shape[-2:]
        center_crop_bbox = _calculate_center_crop_bbox(H, W)
        reference_frame, reference_pose = crop_tensors([reference_frame, reference_pose], center_crop_bbox)

    return image, pose, reference_frame, reference_pose, tgt_face_mask


def crop_around_face(image, bbox, base_scale=0.3, jitter=2.0, aggressive_prob=0.6):

    H, W, _ = image.shape
    if len(bbox) == 0:
        return image

    x1, y1, x2, y2 = bbox
    face_cx = (x1 + x2) / 2
    face_cy = (y1 + y2) / 2
    face_top = min(y1, y2)
    face_bottom = max(y1, y2)
    face_h = face_bottom - face_top

    if random.random() > aggressive_prob:
        aspect_ratio = W / H
        crop_h = int(H * base_scale * random.uniform(1.0, jitter))
        crop_w = int(crop_h * aspect_ratio)

        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)

        top = int(max(0, min(face_cy - crop_h * random.uniform(0.3, 0.5), H - crop_h)))
        left = int(max(0, min(face_cx - crop_w / 2, W - crop_w)))
    else:
        tight_scale = random.uniform(1.5, 2.5) 
        crop_h = int(face_h * tight_scale)
        crop_w = int(crop_h * (W / H)) 

        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)

        top = int(max(0, face_top - 0.15 * crop_h))
        top = min(top, H - crop_h)
        left = int(max(0, face_cx - crop_w / 2))
        left = min(left, W - crop_w)

    cropped = image[top:top + crop_h, left:left + crop_w, :]

    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return image

    if cropped.shape[0] != crop_h or cropped.shape[1] != crop_w:
        return image

    return cropped

# ========================== face region enhancement ========================
L_EYE_IDXS = list(range(36, 42))
R_EYE_IDXS = list(range(42, 48))
NOSE_TIP = 30
MOUTH_L = 48
MOUTH_R = 54
JAW_LINE = list(range(0, 17))

def _to_68x2(arr):
    arr = arr[:1]
    if arr.shape == (1, 68, 2):
        def to_orig(x):
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (68, 2):
                raise ValueError("to_orig expects (68,2)")
            return x[np.newaxis, :, :]
        return arr[0].astype(np.float64), to_orig
    if arr.shape == (68, 2):
        def to_orig(x):
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (68, 2):
                raise ValueError("to_orig expects (68,2)")
            return x
        return arr.astype(np.float64), to_orig
    if arr.shape == (2, 68):
        def to_orig(x):
            x = np.asarray(x, dtype=np.float64)
            if x.shape != (68, 2):
                raise ValueError("to_orig expects (68,2)")
            return x.T
        return arr.T.astype(np.float64), to_orig
    raise ValueError(f"faces shape {arr.shape} not supported; expected (1,68,2) or (68,2) or (2,68)")

def _eye_center(face68, idxs):
    return face68[idxs].mean(axis=0)

def _anchors(face68):
    le = _eye_center(face68, L_EYE_IDXS)
    re = _eye_center(face68, R_EYE_IDXS)
    nose = face68[NOSE_TIP]
    lm = face68[MOUTH_L]
    rm = face68[MOUTH_R]
    if re[0] < le[0]:
        le, re = re, le
    return np.stack([le, re, nose, lm, rm], axis=0)

def _face_scale_only(src68, ref68, target_nose_pos, alpha=1.0, anchor_pairs=[[36, 45], [33, 8]],near_min=0.9, near_max=1.1,
                     near_keep_prob=0.1, push_factor=1.2):

    src = np.asarray(src68, dtype=np.float64)
    ref = np.asarray(ref68, dtype=np.float64)


    center = _anchors(src).mean(axis=0)
    src_centered = src - center

    src_w = np.linalg.norm(src[anchor_pairs[0][0]] - src[anchor_pairs[0][1]])
    ref_w = np.linalg.norm(ref[anchor_pairs[0][0]] - ref[anchor_pairs[0][1]])

    src_h = np.linalg.norm(src[anchor_pairs[1][0]] - src[anchor_pairs[1][1]])
    ref_h = np.linalg.norm(ref[anchor_pairs[1][0]] - ref[anchor_pairs[1][1]])

    scale_x = ref_w / src_w if src_w > 1e-6 else 1.0
    scale_y = ref_h / src_h if src_h > 1e-6 else 1.0

    def push_if_near(val):
        if near_min < val < near_max:
            if np.random.rand() > near_keep_prob:
                if val >= 1.0:
                    return max(near_max, val * push_factor)
                else:
                    return min(near_min, val / push_factor)
        return val

    scale_x = push_if_near(scale_x)
    scale_y = push_if_near(scale_y)

    scaled_local = src_centered.copy()
    scaled_local[:, 0] *= (1 - alpha) + scale_x * alpha
    scaled_local[:, 1] *= (1 - alpha) + scale_y * alpha

    scaled_global = scaled_local + center

    nose_idx = NOSE_TIP
    current_nose = scaled_global[nose_idx]
    offset = target_nose_pos - current_nose
    scaled_global += offset

    return scaled_global


def _face_roundify(src68, ref68, strength=1.0):

    src68 = np.asarray(src68, dtype=np.float64).copy()
    ref68 = np.asarray(ref68, dtype=np.float64)

    src_anchor = _anchors(src68)
    ref_anchor = _anchors(ref68)
    A = []
    B = []
    for sa, ra in zip(ref_anchor, src_anchor):  # 注意顺序：ref -> src
        A.append([sa[0], sa[1], 1, 0, 0, 0])
        A.append([0, 0, 0, sa[0], sa[1], 1])
        B.append(ra[0])
        B.append(ra[1])
    A = np.array(A)
    B = np.array(B)
    x, *_ = np.linalg.lstsq(A, B, rcond=None)
    M_ref2src = np.array([
        [x[0], x[1], x[2]],
        [x[3], x[4], x[5]]
    ])

    ones = np.ones((ref68.shape[0], 1))
    ref68_in_src = np.hstack([ref68, ones]) @ M_ref2src.T

    for idx in JAW_LINE:
        src68[idx] = (1-strength) * src68[idx] + strength * ref68_in_src[idx]

    return src68

def align_poses_by_limb_scaling_only(
        ref_pose, detected_poses,
        limb_change_prob=0.2,       # 骨骼变动的概率（20%）
        face_change_prob=0.8,       # 脸部变动的概率（80%）
        limb_alpha_range=(0.0, 0.8),  # 骨骼缩放强度范围
        face_alpha_range=(0.5, 1.0),  # 脸部缩放强度范围
        min_scale=0.5, max_scale=2.0,
        face_roundify_strength=1.0,
        seed=None):

    if not detected_poses:
        return [], False

    aligned_poses = copy.deepcopy(detected_poses)
    th = 1e-6

    rng = np.random.RandomState(seed) if seed is not None else np.random

    ref_candidate = ref_pose['bodies']['candidate']
    source_candidate_for_ratio = detected_poses[0]['bodies']['candidate']

    def get_limb_length(points, p1_idx, p2_idx):
        return np.linalg.norm(points[p1_idx] - points[p2_idx])

    def make_ratio(p1_idx, p2_idx, coef=1.0, alpha=0.0,
                   near_min=0.9, near_max=1.1,
                   near_keep_prob=0.1, push_factor=1.2):
        len_ref = get_limb_length(ref_candidate, p1_idx, p2_idx)
        len_src = get_limb_length(source_candidate_for_ratio, p1_idx, p2_idx)
        if len_src <= th or len_ref <= th:
            return 1.0

        target = len_ref / len_src
        target = float(np.clip(target, min_scale, max_scale))

        if near_min < target < near_max:
            if np.random.rand() > near_keep_prob:
                if target >= 1.0:
                    target = max(near_max, target * push_factor)
                else:
                    target = min(near_min, target / push_factor)

        return (1.0 * (1.0 - alpha) + target * alpha) * coef

    do_limb_change = (random.random() < limb_change_prob)
    do_face_change = (random.random() < face_change_prob)
    if do_limb_change:
        alpha_limb = rng.uniform(*limb_alpha_range)
        neck_ratio   = make_ratio(0, 1,  alpha=alpha_limb)
        r_eye_ratio  = make_ratio(0,14, alpha=alpha_limb)
        l_eye_ratio  = make_ratio(0,15, alpha=alpha_limb)
        r_ear_ratio  = make_ratio(14,16, alpha=alpha_limb)
        l_ear_ratio  = make_ratio(15,17, alpha=alpha_limb)
    else:
        neck_ratio = r_eye_ratio = l_eye_ratio = r_ear_ratio = l_ear_ratio = 1.0

    has_ref_face = 'faces' in ref_pose and ref_pose['faces'] is not None
    if has_ref_face:
        ref68, _ = _to_68x2(ref_pose['faces'])

    for i, pose in enumerate(aligned_poses):
        candidate = pose['bodies']['candidate']

        if do_limb_change:
            offset = (candidate[1] - candidate[0]) * (1.0 - neck_ratio)
            for idx in [0,14,15,16,17]: candidate[idx] += offset

            offset = (candidate[0] - candidate[14]) * (1.0 - r_eye_ratio)
            for idx in [14,16]: candidate[idx] += offset

            offset = (candidate[0] - candidate[15]) * (1.0 - l_eye_ratio)
            for idx in [15,17]: candidate[idx] += offset

            offset = (candidate[14] - candidate[16]) * (1.0 - r_ear_ratio)
            candidate[16] += offset

            offset = (candidate[15] - candidate[17]) * (1.0 - l_ear_ratio)
            candidate[17] += offset

        if has_ref_face and 'faces' in detected_poses[i] and detected_poses[i]['faces'] is not None:
            
            if do_face_change or do_limb_change:
                try:
                    src68, to_orig = _to_68x2(detected_poses[i]['faces'])
                    face_alpha = rng.uniform(*face_alpha_range)
                    scaled68 = _face_scale_only(src68, ref68, candidate[0], alpha=face_alpha)
                    scaled68 = _face_roundify(scaled68, ref68, strength=face_roundify_strength)
                    pose['faces'] = to_orig(scaled68)
                except Exception as e:
                    print('Face scaling error:', e)

    return aligned_poses, do_limb_change, do_face_change or do_limb_change


def _euclid(p1, p2):
    return np.linalg.norm(p1 - p2)

def align_pose_by_scale_and_translation(ref_pose: dict, 
                        tgt_pose: dict,
                        conf_th: float = 0.5):
    aligned_pose = copy.deepcopy(tgt_pose)
    th = 1e-6

    if 'bodies' not in ref_pose or 'bodies' not in tgt_pose:
        print("Warning: 'bodies' key not found in one of the poses. Returning original target pose.")
        return aligned_pose
        
    ref_kpt = ref_pose['bodies']['candidate'].astype(np.float32)
    tgt_kpt = aligned_pose['bodies']['candidate'].astype(np.float32)
    
    ref_sc = ref_pose['bodies'].get('score', np.ones(ref_kpt.shape[0])).astype(np.float32).reshape(-1)
    tgt_sc = tgt_pose['bodies'].get('score', np.ones(tgt_kpt.shape[0])).astype(np.float32).reshape(-1)


    ref_shoulder_valid = (ref_sc[2] >= conf_th) and (ref_sc[5] >= conf_th)
    tgt_shoulder_valid = (tgt_sc[2] >= conf_th) and (tgt_sc[5] >= conf_th)
    shoulder_ok = ref_shoulder_valid and tgt_shoulder_valid

    ref_hip_valid = (ref_sc[8] >= conf_th) and (ref_sc[11] >= conf_th)
    tgt_hip_valid = (tgt_sc[8] >= conf_th) and (tgt_sc[11] >= conf_th)
    hip_ok = ref_hip_valid and tgt_hip_valid
    
    ref_ear_valid = (ref_sc[16] >= conf_th) and (ref_sc[17] >= conf_th)
    tgt_ear_valid = (tgt_sc[16] >= conf_th) and (tgt_sc[17] >= conf_th)
    ear_ok = ref_ear_valid and tgt_ear_valid

    x_ratio, y_ratio = 1.0, 1.0

    if shoulder_ok and hip_ok:
        # print("Scaling strategy: Using separate X (shoulder) and Y (torso) ratios.")

        ref_shoulder_w = abs(ref_kpt[5, 0] - ref_kpt[2, 0])
        tgt_shoulder_w = abs(tgt_kpt[5, 0] - tgt_kpt[2, 0])
        x_ratio = ref_shoulder_w / tgt_shoulder_w if tgt_shoulder_w > th else 1.0

        ref_torso_h = abs(np.mean(ref_kpt[[8, 11], 1]) - np.mean(ref_kpt[[2, 5], 1]))
        tgt_torso_h = abs(np.mean(tgt_kpt[[8, 11], 1]) - np.mean(tgt_kpt[[2, 5], 1]))
        y_ratio = ref_torso_h / tgt_torso_h if tgt_torso_h > th else 1.0

    elif shoulder_ok:
        # print("Scaling strategy: Fallback to unified ratio using shoulder distance.")
        ref_sh_dist = _euclid(ref_kpt[2], ref_kpt[5])
        tgt_sh_dist = _euclid(tgt_kpt[2], tgt_kpt[5])
        scale_ratio = ref_sh_dist / tgt_sh_dist if tgt_sh_dist > th else 1.0
        x_ratio = y_ratio = scale_ratio
        
    else:
        # print("Scaling strategy: Fallback to unified ratio using ear distance.")
        ref_ear_dist = _euclid(ref_kpt[16], ref_kpt[17])
        tgt_ear_dist = _euclid(tgt_kpt[16], tgt_kpt[17])
        scale_ratio = ref_ear_dist / tgt_ear_dist if tgt_ear_dist > th else 1.0
        x_ratio = y_ratio = scale_ratio
        
    # else:
        # print("Warning: No reliable keypoints for scaling. Using scale=1.0.")

    anchor_idx = 1

    anchor_pt_before_scale = tgt_kpt[anchor_idx].copy()

    def scale(arr):
        if arr is not None and arr.size > 0:
            arr[..., 0] = anchor_pt_before_scale[0] + (arr[..., 0] - anchor_pt_before_scale[0]) * x_ratio
            arr[..., 1] = anchor_pt_before_scale[1] + (arr[..., 1] - anchor_pt_before_scale[1]) * y_ratio

    scale(tgt_kpt)
    scale(aligned_pose.get('faces'))
    scale(aligned_pose.get('hands'))

    offset = ref_kpt[anchor_idx] - tgt_kpt[anchor_idx]
    
    def translate(arr):
        if arr is not None and arr.size > 0:
            arr += offset

    translate(tgt_kpt)
    translate(aligned_pose.get('faces'))
    translate(aligned_pose.get('hands'))
    aligned_pose['bodies']['candidate'] = tgt_kpt
    return aligned_pose
