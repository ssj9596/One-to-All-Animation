import os

import numpy as np
import torch

from .wholebody import Wholebody

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DWposeDetectorAligned:
    def __init__(self, device='cpu'):
        self.pose_estimation = Wholebody()

    def release_memory(self):
        if hasattr(self, 'pose_estimation'):
            del self.pose_estimation
            import gc; gc.collect()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, score = self.pose_estimation(oriImg)
            candidate = candidate[0][np.newaxis, :, :]
            score = score[0][np.newaxis, :]
            nums, _, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:, :18].copy()
            body = body.reshape(nums * 18, locs)
            subset = score[:, :18].copy()
            for i in range(len(subset)):
                for j in range(len(subset[i])):
                    if subset[i][j] > 0.3:
                        subset[i][j] = int(18 * i + j)
                    else:
                        subset[i][j] = -1

            un_visible = score < 0.3
            candidate[un_visible] = -1

            # foot = candidate[:, 18:24]

            faces = candidate[:, 24:92]

            hands = candidate[:, 92:113]
            hands = np.vstack([hands, candidate[:, 113:]])

            faces_score = score[:, 24:92]
            hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

            bodies = dict(candidate=body, subset=subset, score=score[:, :18])
            pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

        return pose


class DWposeDetectorRaw:
    def __init__(self, device='cpu'):
        self.pose_estimation = Wholebody()

    def release_memory(self):
        if hasattr(self, 'pose_estimation'):
            del self.pose_estimation
            import gc; gc.collect()

    def __call__(self, oriImg):
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, score = self.pose_estimation(oriImg)
            pose = dict(candidate=candidate, score=score)
            return pose


dwpose_detector_aligned = DWposeDetectorAligned(device=device)
dwpose_detector_raw = DWposeDetectorRaw(device=device)


def build_pose_dict(pose_raw,
                    img_h, img_w,
                    target_ratio=None,
                    vis_thr=0.3):

    candidate = pose_raw['candidate'][0][np.newaxis, :, :].copy()   # (1,134,2)
    score     = pose_raw['score'][0][np.newaxis, :].copy()          # (1,134)

    nums, _, locs = candidate.shape

    cur_ratio = img_h / img_w

    if (target_ratio is None) or abs(cur_ratio - target_ratio) < 1e-6:
        norm_w, norm_h = img_w, img_h
        pad_x = pad_y  = 0.0
    else:
        if target_ratio > cur_ratio:
            norm_w = img_w
            norm_h = img_w * target_ratio 
            pad_x  = 0.0
            pad_y  = (norm_h - img_h) / 2.0
        else:
            norm_h = img_h
            norm_w = img_h / target_ratio
            pad_y  = 0.0
            pad_x  = (norm_w - img_w) / 2.0

    candidate[:, :, 0] = (candidate[:, :, 0] + pad_x) / float(norm_w)
    candidate[:, :, 1] = (candidate[:, :, 1] + pad_y) / float(norm_h)

    body = candidate[:, :18].copy()
    body = body.reshape(nums * 18, locs)
    subset = score[:, :18].copy()
    for i in range(len(subset)):
        for j in range(len(subset[i])):
            if subset[i][j] > 0.3:
                subset[i][j] = int(18 * i + j)
            else:
                subset[i][j] = -1

    un_visible = score < 0.3
    candidate[un_visible] = -1

    # foot = candidate[:, 18:24]

    faces = candidate[:, 24:92]

    hands = candidate[:, 92:113]
    hands = np.vstack([hands, candidate[:, 113:]])

    faces_score = score[:, 24:92]
    hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

    bodies = dict(candidate=body, subset=subset, score=score[:, :18])
    pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)
    return pose
