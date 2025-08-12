import math, collections, time
import numpy as np
import cv2
from ultralytics import YOLO
from IPython.display import display, clear_output
from ament_index_python.packages import get_package_share_directory
import PIL.Image
import os

# ----------------- Config -----------------
# SOURCE = "output.avi"
PACKAGE_NAME = "pick_and_place_text"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

MODEL = "yolo11n-pose.pt"
YOLO_MODEL_PATH = os.path.join(PACKAGE_PATH, "resource", MODEL)

SOURCE = 0
IMG_SIZE = 640
DEVICE = "cpu"
QUEUE = 15
STABLE_RATIO = 0.7
WAKE_HOLD = 0.8

# 색/스타일
UPPER_COLOR = (0, 255, 255)   # BGR: 노랑 (상체)
LOWER_COLOR = (255, 0, 255)   # BGR: 마젠타 (하체)
JOINT_COLOR = (255, 255, 255) # 관절 점색
LINE_THICK = 3
DOT_RADIUS = 4

# 0:nose,1:leye,2:reye,3:lear,4:rear,5:l_sh,6:r_sh,7:l_elb,8:r_elb,9:l_wri,10:r_wri,
# 11:l_hip,12:r_hip,13:l_knee,14:r_knee,15:l_ank,16:r_ank
NOSE=0; L_SH=5; R_SH=6; L_HIP=11; R_HIP=12; L_KNEE=13; R_KNEE=14; L_ANK=15; R_ANK=16

def midpoint(a, b):
    return ((a[0]+b[0])/2.0, (a[1]+b[1])/2.0)

def angle_to_vertical(p1, p2):
    vx, vy = p2[0]-p1[0], p2[1]-p1[1]
    vnorm = math.hypot(vx, vy)
    if vnorm < 1e-6:
        return None
    cosv = vy / vnorm  # 화면 y축(아래방향)을 수직 벡터로 간주
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def knee_angle(hip, knee, ankle):
    ax, ay = hip[0]-knee[0], hip[1]-knee[1]
    bx, by = ankle[0]-knee[0], ankle[1]-knee[1]
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na*nb < 1e-6:
        return None
    cosv = (ax*bx + ay*by) / (na*nb)
    cosv = max(-1.0, min(1.0, cosv))
    return math.degrees(math.acos(cosv))

def bbox_from_keypoints(kps):
    xs = [x for x,y in kps if x>0 and y>0]
    ys = [y for x,y in kps if x>0 and y>0]
    if not xs or not ys:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    return (x1,y1,x2,y2)


def _line_sample(p1, p2, t=0.5):
    return (p1[0]*(1-t)+p2[0]*t, p1[1]*(1-t)+p2[1]*t)

def extract_line_points(kps, belly_t=0.5):
    """kps: (17,2) -> dict of line skeleton points"""
    def ok(idx): return not (kps[idx][0] == 0 and kps[idx][1] == 0)

    pts = {}
    if ok(NOSE):          pts["head"] = (float(kps[NOSE][0]), float(kps[NOSE][1]))
    if ok(L_SH) and ok(R_SH):
        chest = midpoint(kps[L_SH], kps[R_SH]); pts["chest"] = (float(chest[0]), float(chest[1]))
    if ok(L_HIP) and ok(R_HIP):
        pelvis = midpoint(kps[L_HIP], kps[R_HIP]); pts["pelvis"] = (float(pelvis[0]), float(pelvis[1]))
    if ok(L_KNEE) and ok(R_KNEE):
        knee = midpoint(kps[L_KNEE], kps[R_KNEE]); pts["knee"] = (float(knee[0]), float(knee[1]))
    if ok(L_ANK) and ok(R_ANK):
        foot = midpoint(kps[L_ANK], kps[R_ANK]); pts["foot"] = (float(foot[0]), float(foot[1]))

    # belly: chest~pelvis 선분 내 보간
    if "chest" in pts and "pelvis" in pts:
        belly = _line_sample(pts["chest"], pts["pelvis"], t=belly_t)
        pts["belly"] = (float(belly[0]), float(belly[1]))
    return pts

def draw_color_skeleton(img, pts):
    """상체(head->chest->belly->pelvis) = UPPER_COLOR, 하체(pelvis->knee->foot) = LOWER_COLOR"""
    upper = ["head","chest","belly","pelvis"]
    lower = ["pelvis","knee","foot"]

    # 점 찍기
    for name in set(upper+lower):
        if name in pts:
            cv2.circle(img, (int(pts[name][0]), int(pts[name][1])), DOT_RADIUS, JOINT_COLOR, -1)

    # 선 그리기
    def draw_chain(chain, color):
        chain_xy = [pts[n] for n in chain if n in pts]
        for i in range(len(chain_xy)-1):
            p, q = chain_xy[i], chain_xy[i+1]
            cv2.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, LINE_THICK)

    draw_chain(upper, UPPER_COLOR)
    draw_chain(lower, LOWER_COLOR)
    return img


def classify_pose(kps, img_shape):
    H, W = img_shape[:2]
    pts = kps.astype(float)  # (17,2)

    # 필수 포인트 존재 확인
    def zero(p): return np.all(p==0)
    if zero(pts[L_SH]) or zero(pts[R_SH]) or zero(pts[L_HIP]) or zero(pts[R_HIP]):
        return "unknown", 0.0

    shoulder_mid = midpoint(pts[L_SH], pts[R_SH])
    hip_mid      = midpoint(pts[L_HIP], pts[R_HIP])
    theta_torso  = angle_to_vertical(shoulder_mid, hip_mid)

    bb = bbox_from_keypoints(pts)
    ar = None
    if bb:
        x1,y1,x2,y2 = bb
        w, h = x2-x1, y2-y1
        ar = h / max(w,1e-6)

    # 무릎 각도
    kL = kR = None
    if not zero(pts[L_KNEE]) and not zero(pts[L_ANK]):
        kL = knee_angle(pts[L_HIP], pts[L_KNEE], pts[L_ANK])
    if not zero(pts[R_KNEE]) and not zero(pts[R_ANK]):
        kR = knee_angle(pts[R_HIP], pts[R_KNEE], pts[R_ANK])
    knees = [v for v in [kL,kR] if v is not None]

    # 다리 펴짐 정도(힙-발목 중점 거리 / 키)
    leg_ext = None
    if not zero(pts[L_ANK]) and not zero(pts[R_ANK]) and bb:
        x1,y1,x2,y2 = bb
        h = y2-y1
        ank_mid = midpoint(pts[L_ANK], pts[R_ANK])
        leg_ext = math.hypot(ank_mid[0]-hip_mid[0], ank_mid[1]-hip_mid[1]) / max(h, 1e-6)

    score = {"standing":0.0, "sitting":0.0, "lying":0.0}

    if theta_torso is not None:
        if theta_torso <= 25: score["standing"] += 0.7
        elif theta_torso >= 75: score["lying"] += 0.7
        else: score["sitting"] += 0.6

    if ar is not None:
        if ar > 1.2: score["standing"] += 0.2
        elif ar < 0.8: score["lying"] += 0.2
        else: score["sitting"] += 0.1

    if knees:
        mk = sum(knees)/len(knees)
        if mk >= 165: score["standing"] += 0.15
        elif 60 <= mk <= 140: score["sitting"] += 0.15

    if leg_ext is not None:
        if leg_ext > 0.45: score["standing"] += 0.2
        elif leg_ext < 0.25: score["lying"] += 0.15
        else: score["sitting"] += 0.1

    label = max(score, key=score.get)
    return label, min(1.0, score[label])

# ---------------- Run ----------------
model = YOLO(MODEL)
cap = cv2.VideoCapture(SOURCE)

q = collections.deque(maxlen=QUEUE)
stable_state = "unknown"
prev_stable_state = "unknown"
event_armed = False
event_target = None
event_start_t = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    res = model(frame, imgsz=IMG_SIZE, device=DEVICE, verbose=False)[0]

    if (res.keypoints is None) or (len(res.keypoints) == 0):
        q.append("unknown")
    else:
        # 가장 큰 사람 선택
        if (res.boxes is not None) and (len(res.boxes) > 0):
            areas = (res.boxes.xywh[:,2] * res.boxes.xywh[:,3]).cpu().numpy()
            idx = int(np.argmax(areas))
        else:
            idx = 0

        kps = res.keypoints.xy[idx].cpu().numpy()  # (17,2)
        label, conf = classify_pose(kps, frame.shape)
        q.append(label)

        # ---- 컬러 스켈레톤(상체/하체 색 구분) ----
        pts = extract_line_points(kps, belly_t=0.5)  # 0.33~0.6 등으로 배 위치 조정 가능
        frame = draw_color_skeleton(frame, pts)

    # 상태 안정화 로직
    counts = collections.Counter(q)
    current = counts.most_common(1)[0][0]
    now = time.time()

    if len(q) == QUEUE and counts[current] >= int(STABLE_RATIO*QUEUE) and current != stable_state:
        prev_stable_state = stable_state
        stable_state = current
        if prev_stable_state == "lying" and stable_state in ("sitting","standing"):
            event_armed = True
            event_target = stable_state
            event_start_t = now
        else:
            event_armed = False
            event_target = None
            event_start_t = None

    sit_up_event = False
    stand_up_event = False
    if event_armed and event_target == stable_state and (now - event_start_t >= WAKE_HOLD):
        if event_target == "sitting": sit_up_event = True
        elif event_target == "standing": stand_up_event = True
        event_armed = False

    cv2.putText(frame, f"State: {stable_state}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    if sit_up_event:
        cv2.putText(frame, "SIT UP!", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,165,255), 3)
    if stand_up_event:
        cv2.putText(frame, "STAND UP!", (20,120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

    # ---- Notebook display ----
    clear_output(wait=True)
    display(PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    cv2.imshow("Posture Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
