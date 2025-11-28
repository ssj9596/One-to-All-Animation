import numpy as np
import math
import copy
from wanpose_utils.retarget_pose import get_retarget_pose

L_EYE_IDXS = list(range(36, 42))
R_EYE_IDXS = list(range(42, 48))
NOSE_TIP = 30
MOUTH_L = 48
MOUTH_R = 54
JAW_LINE = list(range(0, 17))


# ===========================Convert wanpose format into our dwpose-like format======================
def aaposemeta_to_dwpose(meta):
    candidate_body = meta['keypoints_body'][:-2][:, :2]
    score_body = meta['keypoints_body'][:-2][:, 2]
    subset_body = np.arange(len(candidate_body), dtype=float)
    subset_body[score_body <= 0] = -1
    bodies = {
        "candidate": candidate_body,
        "subset": np.expand_dims(subset_body, axis=0),   # shape (1, N)
        "score": np.expand_dims(score_body, axis=0)      # shape (1, N)
    }
    hands_coords = np.stack([
        meta['keypoints_right_hand'][:, :2],
        meta['keypoints_left_hand'][:, :2]
    ])
    hands_score = np.stack([
        meta['keypoints_right_hand'][:, 2],
        meta['keypoints_left_hand'][:, 2]
    ])
    faces_coords = np.expand_dims(meta['keypoints_face'][1:][:, :2], axis=0)
    faces_score = np.expand_dims(meta['keypoints_face'][1:][:, 2], axis=0)
    dwpose_format = {
        "bodies": bodies,
        "hands": hands_coords,
        "hands_score": hands_score,
        "faces": faces_coords,
        "faces_score": faces_score
    }
    return dwpose_format

def aaposemeta_obj_to_dwpose(pose_meta: "AAPoseMeta"):
    """
    将 AAPoseMeta 对象转换成 dwpose_like 数据结构
    坐标恢复为相对坐标 (除以 width, height)
    仅处理 None -> 补零
    """
    w = pose_meta.width
    h = pose_meta.height
    
    # 如果是 None，就补成全 0 数组
    def safe(arr, like_shape):
        if arr is None:
            return np.zeros(like_shape, dtype=np.float32)
        arr_np = np.array(arr, dtype=np.float32)
        arr_np = np.nan_to_num(arr_np, nan=0.0)
        return arr_np
    # body
    kps_body = safe(pose_meta.kps_body, (pose_meta.kps_body_p.shape[0], 2))
    candidate_body = kps_body / np.array([w, h])
    score_body = safe(pose_meta.kps_body_p, (candidate_body.shape[0],))
    subset_body = np.arange(len(candidate_body), dtype=float)
    subset_body[score_body <= 0] = -1
    bodies = {
        "candidate": candidate_body,
        "subset": np.expand_dims(subset_body, axis=0),
        "score": np.expand_dims(score_body, axis=0)
    }

    # hands
    kps_rhand = safe(pose_meta.kps_rhand, (pose_meta.kps_rhand_p.shape[0], 2))
    kps_lhand = safe(pose_meta.kps_lhand, (pose_meta.kps_lhand_p.shape[0], 2))
    hands_coords = np.stack([
        kps_rhand / np.array([w, h]),
        kps_lhand / np.array([w, h])
    ])
    hands_score = np.stack([
        safe(pose_meta.kps_rhand_p, (kps_rhand.shape[0],)),
        safe(pose_meta.kps_lhand_p, (kps_lhand.shape[0],))
    ])

    dwpose_format = {
        "bodies": bodies,
        "hands": hands_coords,
        "hands_score": hands_score,
        "faces": None,
        "faces_score": None
    }
    return dwpose_format

# ===============================Face Rough alignment======================

def _to_68x2(arr):
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

def _face_scale_only(src68, ref68, target_nose_pos, alpha=1.0, anchor_pairs=[[36, 45], [27, 8]]):
    """
    粗对齐-根据 ref 的比例调整 src 的脸型，并将鼻尖与 target_nose_pos 对齐。
    anchor_pairs:
      - [36, 45] for x
      - [27, 8] for y
    """
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

    scaled_local = src_centered.copy()
    scaled_local[:, 0] *= (1 - alpha) + scale_x * alpha
    scaled_local[:, 1] *= (1 - alpha) + scale_y * alpha
    scaled_global = scaled_local + center

    nose_idx = NOSE_TIP
    current_nose = scaled_global[nose_idx]
    offset = target_nose_pos - current_nose
    scaled_global += offset

    return scaled_global


# ===============================Select Anchor Pose For Alignment======================

def count_symmetric_pairs(ratios, th=1e-6, ratio_thr=1.5):
    pairs = [
        ('arm3','arm6'),
        ('arm4','arm7'), 
        ('ll1','rl1'),
        ('ll2','rl2')
    ]
    count = 0
    for a, b in pairs:
        v1, v2 = ratios[a], ratios[b]
        if v1 < th or v2 < th:
            continue
        q = v1 / v2
        if 1.0 / ratio_thr <= q <= ratio_thr:
            count += 1
    return count

def pick_good_source_frame(ref_pose, detected_poses, th=1e-6, ratio_thr=1.5,
                           torso_angle_thr_deg=30, horiz_angle_thr_deg=30):
    if not detected_poses:
        return None, -1
    def _angle_with_vertical(v):
        import math
        dx, dy = abs(v[0]), abs(v[1]) + 1e-9
        return math.atan2(dx, dy)
    def _angle_with_horizontal(v):
        import math
        dx, dy = abs(v[0]) + 1e-9, abs(v[1])
        return math.atan2(dy, dx)
    def body_is_upright(cand) -> bool:
        neck = cand[1]; rs = cand[2]; ls = cand[5]
        lh = cand[8]; rh = cand[11]
        for p in (neck, rs, ls, lh, rh):
            if (p <= 0).any():
                return False
        shoulder_c = (ls + rs) / 2
        hip_c = (lh + rh) / 2
        v_torso = hip_c - shoulder_c
        if _angle_with_vertical(v_torso) > math.radians(torso_angle_thr_deg):
            return False
        if _angle_with_horizontal(ls - rs) > math.radians(horiz_angle_thr_deg):
            return False
        if _angle_with_horizontal(lh - rh) > math.radians(horiz_angle_thr_deg):
            return False
        return True

    def limb_len(points, i, j):
        import numpy as np
        return np.linalg.norm(points[i] - points[j])

    def calc_basic_ratios(ref_cand, src_cand):
        def _ratio(i, j):
            l_ref = limb_len(ref_cand, i, j)
            l_src = limb_len(src_cand, i, j)
            return 1.0 if l_src < th else (l_ref / l_src)
        return dict(
            shoulder2=_ratio(1, 2), shoulder5=_ratio(1, 5),
            arm3=_ratio(2, 3), arm6=_ratio(5, 6),
            arm4=_ratio(3, 4), arm7=_ratio(6, 7),
            ll1=_ratio(8, 9), rl1=_ratio(11, 12),
            ll2=_ratio(9, 10), rl2=_ratio(12, 13),
        )

    ref_cand = ref_pose['bodies']['candidate']
    ref_rs = ref_cand[2]
    ref_ls = ref_cand[5]
    ref_neck = ref_cand[1]
    ref_nose = ref_cand[0]
    ref_shoulder_angle = _angle_with_horizontal(ref_ls - ref_rs)
    ref_head_angle = _angle_with_vertical(ref_nose - ref_neck)

    all_candidates = []
    for idx, pose in enumerate(detected_poses):
        cand = pose['bodies']['candidate']
        ratios = calc_basic_ratios(ref_cand, cand)

        shoulder_angle = _angle_with_horizontal(cand[5] - cand[2])
        head_angle = _angle_with_vertical(cand[0] - cand[1])
        angle_score = abs(shoulder_angle - ref_shoulder_angle) + abs(head_angle - ref_head_angle)

        sym_count = count_symmetric_pairs(ratios, th, ratio_thr)
        upright = body_is_upright(cand)

        all_candidates.append({
            "idx": idx,
            "score": angle_score,
            "sym_count": sym_count,
            "upright": upright
        })
        
    sorted_candidates = sorted(
        all_candidates,
        key=lambda x: (-x["sym_count"], not x["upright"], x["score"])
    )

    best = sorted_candidates[0]
    print(f"[pick_good_source_frame] pick frame {best['idx']} "
          f"(sym_count={best['sym_count']}/4, upright={best['upright']}, score={best['score']:.4f})")

    print(f"[pick_good_source_frame] 选中第 {best['idx']} 帧"
          f"(对称匹配={best['sym_count']}/4, 直立={best['upright']}, 角度差={best['score']:.4f})")

    return detected_poses[best["idx"]], best["idx"]

# ===============================Reference Img Pre-Process======================

    
def scale_and_translate_pose(tgt_pose, ref_pose, conf_th=0.9, return_ratio=False):
    aligned_pose = copy.deepcopy(tgt_pose)
    th = 1e-6
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

    if shoulder_ok and hip_ok:
        ref_shoulder_w = abs(ref_kpt[5, 0] - ref_kpt[2, 0])
        tgt_shoulder_w = abs(tgt_kpt[5, 0] - tgt_kpt[2, 0])
        x_ratio = ref_shoulder_w / tgt_shoulder_w if tgt_shoulder_w > th else 1.0

        ref_torso_h = abs(np.mean(ref_kpt[[8, 11], 1]) - np.mean(ref_kpt[[2, 5], 1]))
        tgt_torso_h = abs(np.mean(tgt_kpt[[8, 11], 1]) - np.mean(tgt_kpt[[2, 5], 1]))
        y_ratio = ref_torso_h / tgt_torso_h if tgt_torso_h > th else 1.0
        scale_ratio = (x_ratio + y_ratio) / 2

    elif shoulder_ok:
        ref_sh_dist = np.linalg.norm(ref_kpt[2] - ref_kpt[5])
        tgt_sh_dist = np.linalg.norm(tgt_kpt[2] - tgt_kpt[5])
        scale_ratio = ref_sh_dist / tgt_sh_dist if tgt_sh_dist > th else 1.0
 
    else:
        ref_ear_dist = np.linalg.norm(ref_kpt[16] - ref_kpt[17])
        tgt_ear_dist = np.linalg.norm(tgt_kpt[16] - tgt_kpt[17])
        scale_ratio = ref_ear_dist / tgt_ear_dist if tgt_ear_dist > th else 1.0

    if return_ratio:
        return scale_ratio

    # scale
    anchor_idx = 1
    anchor_pt_before_scale = tgt_kpt[anchor_idx].copy()
    def scale(arr):
        if arr is not None and arr.size > 0:
            arr[..., 0] = anchor_pt_before_scale[0] + (arr[..., 0] - anchor_pt_before_scale[0]) * scale_ratio
            arr[..., 1] = anchor_pt_before_scale[1] + (arr[..., 1] - anchor_pt_before_scale[1]) * scale_ratio
    scale(tgt_kpt)
    scale(aligned_pose.get('faces'))
    scale(aligned_pose.get('hands'))

    # offset
    offset = ref_kpt[anchor_idx] - tgt_kpt[anchor_idx]
    def translate(arr):
        if arr is not None and arr.size > 0:
            arr += offset
    translate(tgt_kpt)
    translate(aligned_pose.get('faces'))
    translate(aligned_pose.get('hands'))
    aligned_pose['bodies']['candidate'] = tgt_kpt

    return aligned_pose, shoulder_ok, hip_ok


# def warp_ref_to_pose(tgt_img,
#                      ref_pose: dict,
#                      tgt_pose: dict,
#                      bg_val=(0, 0, 0),
#                      conf_th=0.8,
#                      align_center=False):

#     def _euclid(p1, p2):
#         return float(np.linalg.norm(p1 - p2))

#     H, W = tgt_img.shape[:2]
#     img_tgt_pose  = draw_pose_aligned(tgt_pose, H, W, without_face=True)

#     ref_kpt = ref_pose['bodies']['candidate'].astype(np.float32)  # (18,2) 0~1
#     tgt_kpt = tgt_pose['bodies']['candidate'].astype(np.float32)
#     ref_sc  = ref_pose['bodies']['score'].astype(np.float32).reshape(-1)      # (18,)
#     tgt_sc  = tgt_pose['bodies']['score'].astype(np.float32).reshape(-1)

#     ref_px = ref_kpt.copy()
#     tgt_px = tgt_kpt.copy()
#     th = 1e-6

#     # ----------  计算 scale ----------
#     shoulder_ok = (tgt_sc[2] >= conf_th) & (tgt_sc[5] >= conf_th) & (ref_sc[2] >= conf_th) & (ref_sc[5] >= conf_th)
#     hip_ok      = (tgt_sc[8] >= conf_th) & (tgt_sc[11] >= conf_th) & (ref_sc[8] >= conf_th) & (ref_sc[11] >= conf_th)

#     ref_sh = _euclid(ref_px[2], ref_px[5]);  tgt_sh = _euclid(tgt_px[2], tgt_px[5])
#     ref_torso = _euclid(0.5*(ref_px[2]+ref_px[5]), 0.5*(ref_px[8]+ref_px[11]))
#     tgt_torso = _euclid(0.5*(tgt_px[2]+tgt_px[5]), 0.5*(tgt_px[8]+tgt_px[11]))
#     ref_ear = _euclid(ref_px[16], ref_px[17]); tgt_ear = _euclid(tgt_px[16], tgt_px[17])

#     # scale_ratio = 1.0
#     if shoulder_ok and hip_ok and min(ref_sh, tgt_sh, ref_torso, tgt_torso) > th:
#         scale_ratio = 0.5 * (ref_sh / tgt_sh + ref_torso / tgt_torso)
#         print("shoulder_ok and hip_ok",scale_ratio)

#     elif shoulder_ok and min(ref_sh, tgt_sh) > th:
#         scale_ratio = ref_sh / tgt_sh
#         print("shoulder_ok",scale_ratio)

#     else:
#         scale_ratio = ref_ear / tgt_ear
#         print("ear",scale_ratio)

#     anchor = 1
#     x0 = tgt_px[anchor][0] * W
#     y0 = tgt_px[anchor][1] * H
#     ref_x = ref_px[anchor][0] * W if not align_center else W/2
#     ref_y = ref_px[anchor][1] * H 
#     dx = ref_x - x0
#     dy = ref_y - y0 

#     # 仿射变换
#     M = np.array([[scale_ratio, 0, (1-scale_ratio)*x0 + dx],
#                   [0, scale_ratio, (1-scale_ratio)*y0 + dy]],
#                  dtype=np.float32)

#     img_warp = cv2.warpAffine(tgt_img, M, (W, H),
#                               flags=cv2.INTER_LINEAR,
#                               borderValue=bg_val)

#     img_tgt_pose_warp = cv2.warpAffine(img_tgt_pose, M, (W, H),
#                               flags=cv2.INTER_LINEAR,
#                               borderValue=bg_val)


#     zeros = np.zeros((H, W), dtype=np.uint8)
#     mask_warp = cv2.warpAffine(zeros, M, (W, H),
#                                flags=cv2.INTER_NEAREST,
#                                borderValue=1)
#     # pre-processed reference img | reference pose | mask 
#     return img_warp, img_tgt_pose_warp,  mask_warp

# ===============================Align to Ref Driven Pose Retarget ======================

def align_to_reference(ref_pose_meta, tpl_pose_metas, tpl_dwposes, anchor_idx=None):
    # pose retarget + face rough align
    
    ref_pose_dw = aaposemeta_to_dwpose(ref_pose_meta)
    if anchor_idx is None:
        _, best_idx = pick_good_source_frame(ref_pose_dw, tpl_dwposes)
    else:
        best_idx = anchor_idx
    tpl_pose_meta_best = tpl_pose_metas[best_idx]

    tpl_retarget_pose_metas = get_retarget_pose(
        tpl_pose_meta_best,
        ref_pose_meta,
        tpl_pose_metas,
        None, None
    )

    retarget_dwposes = [aaposemeta_obj_to_dwpose(pm) for pm in tpl_retarget_pose_metas]

    if ref_pose_dw['faces'] is not None:
        ref68, _ = _to_68x2(ref_pose_dw['faces'])
        for frame_idx, (tpl_dw, rt_dw) in enumerate(zip(tpl_dwposes, retarget_dwposes)):
            if tpl_dw['faces'] is None:
                continue
            src68, to_orig = _to_68x2(tpl_dw['faces'])
            target_nose_pos = rt_dw['bodies']['candidate'][0]
            scaled68 = _face_scale_only(src68, ref68, target_nose_pos, alpha=1.0)
            rt_dw['faces'] = to_orig(scaled68)
            rt_dw['faces_score'] = tpl_dw['faces_score']

    return retarget_dwposes

# ===============================Rescale-Ref && Change part of pose(Option)======================


def compute_ratios_stepwise(ref_scores, source_scores, ref_pts, src_pts, conf_th=0.9, th=1e-6):

    def keypoint_valid(idx):
        return ref_scores[0, idx] >= conf_th and source_scores[0, idx] >= conf_th

    def safe_ratio(p1, p2):
        len_ref = np.linalg.norm(ref_pts[p1] - ref_pts[p2])
        len_src = np.linalg.norm(src_pts[p1] - src_pts[p2])
        if len_src > th:
            return len_ref / len_src
        else:
            return 1.0

    ratio_pairs = [
        (0,1),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),
        (0,14),(0,15),(14,16),(15,17),
        (8,9),(9,10),(11,12),(12,13),
        (1,8),(1,11)
    ]
    ratios = {p: 1.0 for p in ratio_pairs}

    parent_map = {
        (3, 4): (2, 3),   
        (6, 7): (5, 6),   
        (9, 10): (8, 9),  
        (12, 13): (11, 12)
    }

    # Group 1 — head only
    if all(keypoint_valid(i) for i in [0,1,14,15,16,17]):
        ratios[(0,1)]  = safe_ratio(0,1)
        ratios[(0,14)] = safe_ratio(0,14)
        ratios[(0,15)] = safe_ratio(0,15)
        ratios[(14,16)]= safe_ratio(14,16)
        ratios[(15,17)]= safe_ratio(15,17)

    # Group 2 — +shoulder
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17]):
        ratios[(1,2)] = safe_ratio(1,2)
        ratios[(1,5)] = safe_ratio(1,5)
    
    # Group 3 — +upper arm
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17,3,6]):
        ratios[(2,3)] = safe_ratio(2,3)
        ratios[(5,6)] = safe_ratio(5,6)
        ratios[(3,4)] = ratios[parent_map[(3,4)]]
        ratios[(6,7)] = ratios[parent_map[(6,7)]]
    
    # Group 4 — +hips
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17,3,6,8,11]):
        ratios[(1,8)] = safe_ratio(1,8)
        ratios[(1,11)] = safe_ratio(1,11)

    # Group 5 — forearm own
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17,3,6,8,11,4,7]):
        ratios[(3,4)] = safe_ratio(3,4)
        ratios[(6,7)] = safe_ratio(6,7)

    # Group 6 — knees
    if all(keypoint_valid(i) for i in [0,1,2,5,14,15,16,17,3,6,8,11,4,7,9,12]):
        ratios[(8,9)] = safe_ratio(8,9)
        ratios[(11,12)] = safe_ratio(11,12)
        ratios[(9,10)] = ratios[parent_map[(9,10)]]
        ratios[(12,13)]= ratios[parent_map[(12,13)]]

    # Full body — all ratios
    if all(keypoint_valid(i) for i in range(18)):
        for p in ratio_pairs:
            ratios[p] = safe_ratio(*p)

    symmetric_pairs = [
        ((1, 2), (1, 5)),    # 两肩
        ((2, 3), (5, 6)),    # 上臂
        ((3, 4), (6, 7)),    # 前臂
        ((8, 9), (11, 12)),  # 大腿
        ((9, 10), (12, 13))  # 小腿
    ]
    for left_key, right_key in symmetric_pairs:
        left_val = ratios.get(left_key)
        right_val = ratios.get(right_key)
        if left_val is not None and right_val is not None:
            avg_val = (left_val + right_val) / 2.0
            ratios[left_key] = avg_val
            ratios[right_key] = avg_val

    eye_pairs = [
        ((13, 15), (14, 16))
    ]
    for left_key, right_key in eye_pairs:
        left_val = ratios.get(left_key)
        right_val = ratios.get(right_key)
        if left_val is not None and right_val is not None:
            avg_val = (left_val + right_val) / 2.0
            ratios[left_key] = avg_val
            ratios[right_key] = avg_val

    return ratios

def align_to_pose(ref_dwpose, tpl_dwposes,anchor_idx=None,conf_th=0.9,):
    detected_poses = copy.deepcopy(tpl_dwposes)

    th = 1e-6
    if anchor_idx is None:
        best_pose, _ = pick_good_source_frame(ref_dwpose, tpl_dwposes)
    else:
        best_pose = tpl_dwposes[anchor_idx]
    ref_pose_scaled, _, _ = scale_and_translate_pose(ref_dwpose, best_pose, conf_th=conf_th)

    ref_candidate = ref_pose_scaled['bodies']['candidate'].astype(np.float32)
    ref_scores    = ref_pose_scaled['bodies']['score'].astype(np.float32)

    source_candidate = best_pose['bodies']['candidate'].astype(np.float32)
    source_scores = best_pose['bodies']['score'].astype(np.float32)

    has_ref_face = 'faces' in ref_pose_scaled and ref_pose_scaled['faces'] is not None and ref_pose_scaled['faces'].size > 0
    if has_ref_face:
        try:
            ref68, _ = _to_68x2(ref_pose_scaled['faces'])
        except Exception as e:
            print("参考脸转换失败:", e)
            has_ref_face = False
     
    ratios = compute_ratios_stepwise(ref_scores, source_scores, ref_candidate, source_candidate, conf_th=conf_th, th=1e-6)

    for pose in detected_poses:
        candidate = pose['bodies']['candidate']
        faces = pose['faces']
        hands = pose['hands']

        # ===== Neck =====
        ratio = ratios[(0, 1)]
        x_offset = (candidate[1][0] - candidate[0][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[0][1]) * (1. - ratio)
        candidate[[0, 14, 15, 16, 17], 0] += x_offset
        candidate[[0, 14, 15, 16, 17], 1] += y_offset

        # ===== Shoulder Right =====
        ratio = ratios[(1, 2)]
        x_offset = (candidate[1][0] - candidate[2][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[2][1]) * (1. - ratio)
        candidate[[2, 3, 4], 0] += x_offset
        candidate[[2, 3, 4], 1] += y_offset
        hands[1, :, 0] += x_offset
        hands[1, :, 1] += y_offset

        # ===== Shoulder Left =====
        ratio = ratios[(1, 5)]
        x_offset = (candidate[1][0] - candidate[5][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[5][1]) * (1. - ratio)
        candidate[[5, 6, 7], 0] += x_offset
        candidate[[5, 6, 7], 1] += y_offset
        hands[0, :, 0] += x_offset
        hands[0, :, 1] += y_offset

        # ===== Upper Arm Right =====
        ratio = ratios[(2, 3)]
        x_offset = (candidate[2][0] - candidate[3][0]) * (1. - ratio)
        y_offset = (candidate[2][1] - candidate[3][1]) * (1. - ratio)
        candidate[[3, 4], 0] += x_offset
        candidate[[3, 4], 1] += y_offset
        hands[1, :, 0] += x_offset
        hands[1, :, 1] += y_offset

        # ===== Forearm Right =====
        ratio = ratios[(3, 4)]
        x_offset = (candidate[3][0] - candidate[4][0]) * (1. - ratio)
        y_offset = (candidate[3][1] - candidate[4][1]) * (1. - ratio)
        candidate[4, 0] += x_offset
        candidate[4, 1] += y_offset
        hands[1, :, 0] += x_offset
        hands[1, :, 1] += y_offset

        # ===== Upper Arm Left =====
        ratio = ratios[(5, 6)]
        x_offset = (candidate[5][0] - candidate[6][0]) * (1. - ratio)
        y_offset = (candidate[5][1] - candidate[6][1]) * (1. - ratio)
        candidate[[6, 7], 0] += x_offset
        candidate[[6, 7], 1] += y_offset
        hands[0, :, 0] += x_offset
        hands[0, :, 1] += y_offset

        # ===== Forearm Left =====
        ratio = ratios[(6, 7)]
        x_offset = (candidate[6][0] - candidate[7][0]) * (1. - ratio)
        y_offset = (candidate[6][1] - candidate[7][1]) * (1. - ratio)
        candidate[7, 0] += x_offset
        candidate[7, 1] += y_offset
        hands[0, :, 0] += x_offset
        hands[0, :, 1] += y_offset

        # ===== Head parts =====
        for (p1, p2) in [(0,14),(0,15),(14,16),(15,17)]:
            ratio = ratios[(p1,p2)]
            x_offset = (candidate[p1][0] - candidate[p2][0]) * (1. - ratio)
            y_offset = (candidate[p1][1] - candidate[p2][1]) * (1. - ratio)
            candidate[p2, 0] += x_offset
            candidate[p2, 1] += y_offset

        # ===== Hips (added) =====
        ratio = ratios[(1, 8)]
        x_offset = (candidate[1][0] - candidate[8][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[8][1]) * (1. - ratio)
        candidate[8, 0] += x_offset
        candidate[8, 1] += y_offset

        ratio = ratios[(1, 11)]
        x_offset = (candidate[1][0] - candidate[11][0]) * (1. - ratio)
        y_offset = (candidate[1][1] - candidate[11][1]) * (1. - ratio)
        candidate[11, 0] += x_offset
        candidate[11, 1] += y_offset

        # ===== Legs =====
        ratio = ratios[(8, 9)]
        x_offset = (candidate[9][0] - candidate[8][0]) * (ratio - 1.)
        y_offset = (candidate[9][1] - candidate[8][1]) * (ratio - 1.)
        candidate[[9, 10], 0] += x_offset
        candidate[[9, 10], 1] += y_offset

        ratio = ratios[(9, 10)]
        x_offset = (candidate[10][0] - candidate[9][0]) * (ratio - 1.)
        y_offset = (candidate[10][1] - candidate[9][1]) * (ratio - 1.)
        candidate[10, 0] += x_offset
        candidate[10, 1] += y_offset

        ratio = ratios[(11, 12)]
        x_offset = (candidate[12][0] - candidate[11][0]) * (ratio - 1.)
        y_offset = (candidate[12][1] - candidate[11][1]) * (ratio - 1.)
        candidate[[12, 13], 0] += x_offset
        candidate[[12, 13], 1] += y_offset

        ratio = ratios[(12, 13)]
        x_offset = (candidate[13][0] - candidate[12][0]) * (ratio - 1.)
        y_offset = (candidate[13][1] - candidate[12][1]) * (ratio - 1.)
        candidate[13, 0] += x_offset
        candidate[13, 1] += y_offset

        # rough align
        if has_ref_face and 'faces' in pose and pose['faces'] is not None and pose['faces'].size > 0:
            try:
                src68, to_orig = _to_68x2(pose['faces'])
                scaled68 = _face_scale_only(src68, ref68, candidate[0], alpha=1.0)
                pose['faces'] = to_orig(scaled68)
            except Exception as e:
                print("换脸失败:", e)
                continue
            
    return detected_poses






