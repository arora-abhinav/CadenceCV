#This file will be used to write the inference metrics obtained from svm_inference.py. 
import numpy as np
import scipy.signal
import cv2
import torch
import sys
import matplotlib.pyplot as plt
from enum import Enum
import json
sys.path.append("/Users/abhinavarora/Desktop/CadenceCV/ml/src")
from utils.numpy_encoder import NumpyEncoder
from svm_inference import d_frame_preds_legs_indices, u_frame_preds_legs_indices, cap, valid_frames, non_normal_kpts

class unit(Enum):
    M = 1
    CM = 2
    MM = 3

class overstride(Enum):
    NEUTRAL = 1
    MILD = 2
    SEVERE = 3

class strikefoot_type(Enum):
    HEEL = 1
    MIDFOOT = 2
    FOREFOOT = 3

#Diagnostic (a): plot the raw per-frame knee flexion curve for each leg with the DETECTED strikes marked.
#What im looking for: at a true touchdown the knee should be near a LOCAL LOW (leg reaching out, fairly
#straight), then flex through stance. If the red strike dots land partway UP the rise, the strike frame is
#late -> inflated contact flexion. If a dot lands on a big ~90 deg swing PEAK, thats a leg SWITCH (im reading
#the other leg mid-swing), which would also blow up the numbers.
def diagnose_knee_flexion_timing(detected_frames, strikes_left, strikes_right,
                                 left_hip_arr, left_knee_arr, left_ankle_arr,
                                 right_hip_arr, right_knee_arr, right_ankle_arr):
    np_det = np.array(detected_frames)

    def flex_curve(hip_arr, knee_arr, ankle_arr):
        #same 180 - interior-angle convention the metrics use, per frame
        curve = []
        for h, k, a in zip(hip_arr, knee_arr, ankle_arr):
            h, k, a = np.array(h), np.array(k), np.array(a)
            ba, bc = h - k, a - k
            cos = np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)
            curve.append(180 - np.rad2deg(np.arccos(cos)))
        return np.array(curve)

    def strike_positions(strikes):
        pos = []
        for f in strikes:
            where = np.where(np_det == f[0])[0]
            if len(where):
                pos.append(where[0])
        return pos

    left_curve = flex_curve(left_hip_arr, left_knee_arr, left_ankle_arr)
    right_curve = flex_curve(right_hip_arr, right_knee_arr, right_ankle_arr)
    l_pos, r_pos = strike_positions(strikes_left), strike_positions(strikes_right)

    #distance between the two hip keypoints (11 vs 12). in a side view they nearly overlap, so a sudden SPIKE
    #= a detection glitch / the far hip jumping - a frame i shouldnt trust. log scale so tiny baseline stays
    #readable while instantaneous spikes pop. floor at a tiny epsilon so log(0) doesnt blow up.
    hip_dist = np.array([np.linalg.norm(np.array(l) - np.array(r)) for l, r in zip(left_hip_arr, right_hip_arr)])
    hip_dist = np.maximum(hip_dist, 1e-6)

    fig, (axl, axr, axh) = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    axl.plot(np_det, left_curve, color="tab:blue", lw=1.2, label="left knee flexion")
    axl.scatter(np_det[l_pos], left_curve[l_pos], color="red", s=60, zorder=5, label="detected strike")
    axl.set_ylabel("left knee flexion (deg)"); axl.legend(loc="upper right")
    axl.set_title("Diagnostic (a): knee flexion vs frame, strikes marked  "
                  "(strike should sit at a LOW; high dot = late strike / leg switch)")
    axr.plot(np_det, right_curve, color="tab:orange", lw=1.2, label="right knee flexion")
    axr.scatter(np_det[r_pos], right_curve[r_pos], color="red", s=60, zorder=5, label="detected strike")
    axr.set_ylabel("right knee flexion (deg)"); axr.legend(loc="upper right")
    axh.plot(np_det, hip_dist, color="tab:green", lw=1.2, label="|left hip - right hip|")
    axh.set_yscale("log")
    axh.set_ylabel("hip-hip dist (log)"); axh.set_xlabel("frame"); axh.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


#Clauclating the average time in air per gait cycle and average time in ground contact
#kpts is NOT normalised
def calculate_metrics(strikefoot_frames_dict:dict, toe_off_frames_dict:dict, detected_frames, cap: cv2.VideoCapture, kpts:torch.Tensor, shin_bone_measurement, height, mass=None):
    """
    shin_bone_measurement and height MUST be in meters
    mass is in kg and is optional - its only needed for the stiffness/force stuff (Morin's method). Everything
    else works without it since i kept those calibration-free on purpose
    """

    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS)
    flight_times = []
    gct_times = []

    strikefoot_frames = np.array([x for x in strikefoot_frames_dict.keys()])
    toe_off_frames = np.array([x for x in toe_off_frames_dict.keys()])

    strikefoot_frames_left, strikefoot_frames_right = [(x, "D") for x in strikefoot_frames if strikefoot_frames_dict[x] == "l"], [(x, "D") for x in strikefoot_frames if strikefoot_frames_dict[x] == "r"]
    toe_off_frames_left, toe_off_frames_right = [(x, "U") for x in toe_off_frames if toe_off_frames_dict[x] == "l"], [(x, "U") for x in toe_off_frames if toe_off_frames_dict[x] == "r"]
    gait_cycle_left = toe_off_frames_left + strikefoot_frames_left
    gait_cycle_right = toe_off_frames_right + strikefoot_frames_right
    gait_cycle_left.sort()
    gait_cycle_right.sort()

    #Due to YOLO missing out on some frames, the actual frame count is required so that the ratio of
    #detected:actual frame count can be taken
    actual_frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    detected_frame_count = len(detected_frames)

    #Cadence is the steps per second
    cadence = len(strikefoot_frames) / duration * (detected_frame_count/int(actual_frame_count))

    def smooth_keypoints(kpts, strikefoot_frames):
        k = np.array(kpts.tolist()) if hasattr(kpts, "tolist") else np.array(kpts, dtype=float)
        gaps = np.diff(np.sort(strikefoot_frames))
        median_gap = np.median(gaps) if len(gaps) else len(k)
        window = int((1/3) * median_gap)
        if window % 2 == 0:
            window += 1
        window = max(window, 5)
        if window > len(k):
            window = len(k) if len(k) % 2 == 1 else len(k) - 1
        return scipy.signal.savgol_filter(k, window_length=window, polyorder=3, axis=0)

    kpts = smooth_keypoints(kpts, strikefoot_frames)

    right_hip_arr = []
    left_hip_arr = []
    left_ankle_arr = []
    right_ankle_arr = []
    left_knee_arr = []
    right_knee_arr = []
    left_heel_arr = []
    right_heel_arr = []
    left_shoulder_arr, right_shoulder_arr = [], []
    left_toe_kpt_arr, right_toe_kpt_arr = [], []
    left_heel_kpt_arr, right_heel_kpt_arr = [], []
    for k in kpts.tolist():
        left_hip_arr.append(k[11])
        right_hip_arr.append(k[12])
        left_knee_arr.append(k[13])
        right_knee_arr.append(k[14])
        left_ankle_arr.append(k[15])
        right_ankle_arr.append(k[16])
        left_heel_arr.append(k[19])
        right_heel_arr.append(k[20])
        left_shoulder_arr.append(k[5])
        right_shoulder_arr.append(k[6])
        left_heel_kpt_arr.append(k[17])
        right_heel_kpt_arr.append(k[18])
        left_toe_kpt_arr.append(k[19])
        right_toe_kpt_arr.append(k[20])

    fps = actual_frame_count / duration
    
    np_strikefoot_frames = np.array(strikefoot_frames)
    np_toeoff_frames = np.array(toe_off_frames)
    for takeoff in toe_off_frames:
        #Find the next landing after this takeoff. Finding all landings initially to check if there even is a landing 
        #after the take off frames
        next_landings = np_strikefoot_frames[np_strikefoot_frames > takeoff]
        if len(next_landings) > 0:
            next_landing = next_landings[0]
            flight_times.append(next_landing - takeoff)
    
    for landing in strikefoot_frames:
        next_takeoffs = np_toeoff_frames[np_toeoff_frames > landing]
        if len(next_takeoffs) > 0:
            next_takeoff = next_takeoffs[0]  
            gct_times.append(next_takeoff - landing)
    #Divide by detected frame count to obtain metrics in seconds
    average_flight_time = (np.mean(flight_times) if flight_times else 0)/fps
    average_gct = (np.mean(gct_times) if gct_times else 0)/fps

    #Vertical oscillation: 
    # 1) Calculate the hip's mean Y coordinate (mean of right and left hip Y) over all frames
    # 2) Apply the Savitz-Golay smoothing filter (finetune the window size)
    # 3) Separate out the smoothed array based on each gait cycle: (continuous pairs of 0s and 1s)
    # 4) Obtain the vertical difference between each peak and trough (since pixel coordinates go downward)

    # Index 1 is the Y coordinate (vertical axis in pixel space), Index 0 is the X coordinate
    n = min(len(left_hip_arr), len(right_hip_arr))
    mean_hip_x_coords, mean_hip_y_coords = [], []
    for i in range(n):
        mean_hip_y_coords.append(np.mean([left_hip_arr[i][1], right_hip_arr[i][1]]))
        mean_hip_x_coords.append(np.mean([left_hip_arr[i][0], right_hip_arr[i][0]]))
    
    #The window length is the number of coefficients for smoothing
    #The poly order is the degree of the filtering polynomial
    #Finetune these accordingly
    mean_hip_y_coords = scipy.signal.savgol_filter(mean_hip_y_coords, window_length=4, polyorder=3)

    #Extracting gait cycles: A gait cycle is essentially (for the same leg the strikefoot -> toe off -> strikefoot) again
    mask = ["D", "U", "D"]
    def extract_cycles(gait_cycle):
        i = 0
        res = []
        while i < len(gait_cycle) - 2:
            cycle = [gait_cycle[i][1], gait_cycle[i+1][1], gait_cycle[i+2][1]]
            if cycle == mask:
                res.append([gait_cycle[i][0], gait_cycle[i+2][0]])
            i += 1

        return res

    extracted_left_gait_cycles = extract_cycles(gait_cycle_left)
    extracted_right_gait_cycles = extract_cycles(gait_cycle_right)
    np_detected = np.array(detected_frames)
    def extract_oscillation(gait_cycle):
        oscillations = []
        for start, end in gait_cycle:
            #Finding the exact index where the start and end frames are present in detected frames. using numpy for quicker indexing
            start_pos = np.array(np.where(np_detected == start)).flatten().tolist()[0]
            end_pos = np.array(np.where(np_detected == end)).flatten().tolist()[0]
            coords = np.array(mean_hip_y_coords)[start_pos: end_pos + 1]
            peak = max(coords)
            trough = min(coords)
            oscillations.append(peak - trough)

        return oscillations

    avg_left_side_oscillation = np.mean(np.array(extract_oscillation(extracted_left_gait_cycles)))
    avg_right_side_oscillation = np.mean(np.array(extract_oscillation(extracted_right_gait_cycles)))
    avg_total_oscillation = np.mean([avg_left_side_oscillation, avg_right_side_oscillation])

    #Calculating stride length requires a normalisation factor (for example the measurement of the shin bone)
    #If the video is truly sideview, then the measurement of the shin should not change at all. 
    # However, the average length of the shin bone across all frames will be taken for best video calibration
    def extract_ratio(measurement_unit:unit):
        #Complicated af statement lol

        avg_shin_bone_length = np.mean(np.sqrt(np.square((np.array(left_knee_arr)[:,0] - np.array(left_ankle_arr)[:,0])) + np.square((np.array(left_knee_arr)[:,1] - np.array(left_ankle_arr)[:,1]))))
        if measurement_unit == unit.MM:
            pixel_ratio = avg_shin_bone_length/(shin_bone_measurement * 1000)
        elif measurement_unit == unit.M:
            pixel_ratio = avg_shin_bone_length/shin_bone_measurement
        elif measurement_unit == unit.CM:
            pixel_ratio = avg_shin_bone_length/(shin_bone_measurement * 100)

        return pixel_ratio

    pixel_to_meter_ratio = extract_ratio(unit.M)
    pixel_to_cm_ratio = extract_ratio(unit.CM)

    def extract_stride_length(gait_cycle, ankle_arr):
        #Extracting stride length based on gait_cycles
        stride_lengths = []
        for start, end in gait_cycle:
            start_pos = np.array(np.where(np_detected == start)).flatten().tolist()[0]
            end_pos = np.array(np.where(np_detected == end)).flatten().tolist()[0]
            ankle_cycle = np.array(ankle_arr)[start_pos: end_pos + 1]
            peak = max(ankle_cycle[:,0])
            trough = max(ankle_cycle[:,0])
            stride_lengths.append(peak - trough)

        return stride_lengths

    avg_left_stride_length = (np.mean(np.array(extract_stride_length(extracted_left_gait_cycles, left_ankle_arr)))) * (1/pixel_to_meter_ratio)
    avg_right_stride_length = (np.mean(np.array(extract_stride_length(extracted_right_gait_cycles, right_ankle_arr)))) * (1/pixel_to_meter_ratio)
    avg_overall_stride_length = (np.mean([avg_left_stride_length, avg_left_stride_length])) * pixel_to_meter_ratio

    #Classifying overstrides: 
    # a) Shin Test: Checking the angle between the vertical and the shin bone. (ankle to knee). If angle > 5 degrees, then the person is overstriding
    # b) Center of Mass test: Seeing if the ankle goes beyond the Center of Mass (mean of the hip's coordinates):
    # 1) 0 - 5 cm beyond the COM: Optimal
    # 2) 5 - 10 cm beyond the COM: Mild Overstriding
    # 3) 10 - 15 cm beyond the COM: Severe Overstriding

    #Uses of both tests: shin test is to see injury risk, whereas COM is actually used to test force production. 

    OFFSET_NEUTRAL_MAX_PX = (5 * pixel_to_meter_ratio)
    OFFSET_MILD_MAX_PX = (5 * pixel_to_meter_ratio)
    TIBIAL_NEUTRAL_MAX_RAD = 5 
    TIBIAL_MILD_MAX_RAD = 10

    def _classify_offset(offset_px):
        if offset_px < OFFSET_NEUTRAL_MAX_PX:
            return overstride.NEUTRAL
        elif offset_px < OFFSET_MILD_MAX_PX:
            return overstride.MILD
        else:
            return overstride.SEVERE

    def _classify_tibial(tibial_rad):
        # abs() because the angle can be negative depending on running direction
        angle = abs(tibial_rad)
        if angle < TIBIAL_NEUTRAL_MAX_RAD:
            return overstride.NEUTRAL
        elif angle < TIBIAL_MILD_MAX_RAD:
            return overstride.MILD
        else:
            return overstride.SEVERE

    def calculate_tibial(shin_vector):
        angle = np.arctan2(shin_vector[1], shin_vector[0])
        angle = np.rad2deg(np.pi/2 - angle)
        return angle

    def overstriding_classification(strikefoot_frames, heel_x_coords, knee_arr, ankle_arr):
        overstriding_COM_mild = []
        overstriding_COM_severe= []
        overstriding_shin_mild = []
        overstriding_shin_severe = []
        for frame in strikefoot_frames:
            #frame[0] is taken because its in the format of (frame, "D") wjere "D" signifies its a strikeoot frame (stupid system, will change later)
            ind = np.array(np.where(np_detected == frame[0])).flatten().tolist()[0]
            hip_x_c = mean_hip_x_coords[ind]
            heel_x_c = heel_x_coords[ind]
            COM_res = _classify_offset(np.abs(hip_x_c - heel_x_c))
            if COM_res == overstride.MILD:
                overstriding_COM_mild.append(frame)
            elif COM_res == overstride.SEVERE:
                overstriding_COM_severe.append(frame)

            shin_vector = np.array(knee_arr[ind]) - np.array(ankle_arr[ind])
            tibial_angle = calculate_tibial(shin_vector)
            shin_res = _classify_tibial(tibial_angle)
            if shin_res == overstride.MILD:
                overstriding_shin_mild.append(frame)
            elif shin_res == overstride.SEVERE:
                overstriding_shin_severe.append(frame)

        return {"COM Mild": overstriding_COM_mild,
                "COM Severe": overstriding_COM_severe,
                "Shin Mild": overstriding_shin_mild,
                "Shin Severe": overstriding_shin_severe}

    left_foot_overstrides = overstriding_classification(strikefoot_frames_left, np.array(left_heel_arr)[:,0], left_knee_arr, left_ankle_arr)
    right_foot_overstrides = overstriding_classification(strikefoot_frames_right, np.array(right_heel_arr)[:,0], right_knee_arr, right_ankle_arr)

    #Generic joint angle at vertex b for the points a-b-c. 180 = dead straight limb, smaller = more bent.
    #Wrote this once since im reusing it for the knee, hip drive
    def joint_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        #Clipping because float error can nudge this a hair past +-1 and then arccos just NaNs on me
        cos = np.clip(cos, -1.0, 1.0)
        return np.rad2deg(np.arccos(cos))

    #Tiny helper: give it a real frame number, get back where it sits in detected_frames so i can index the arrays
    def frame_to_index(frame):
        return np.array(np.where(np_detected == frame)).flatten().tolist()[0]

    #Stride length in FRAMES here (strikefoot -> same leg strikefoot).
    all_gait_cycles = extracted_left_gait_cycles + extracted_right_gait_cycles
    avg_stride_frames = np.mean([end - start for start, end in all_gait_cycles])
    avg_gct_frames = np.mean(gct_times) if gct_times else 0
    avg_aerial_frames = np.mean(flight_times) if flight_times else 0

    #Swing = the whole stride minus the bit youre on the ground. Different from aerial time (aerial = BOTH feet off)
    avg_swing_frames = avg_stride_frames - avg_gct_frames

    #Duty factor = fraction of the stride youre actually planted. Low = springy/floaty runner, high = grinding into
    #the ground.
    duty_factor = avg_gct_frames / avg_stride_frames

    #Need real seconds for the physics below (Morin's eqns dont care about pixels but they very much care about time)
    seconds_per_frame = 1 / fps
    t_c = avg_gct_frames * seconds_per_frame         
    t_a = avg_aerial_frames * seconds_per_frame    
    stride_time = avg_stride_frames * seconds_per_frame
    swing_time = avg_swing_frames * seconds_per_frame

    #Speed = how far one stride carries you over how long it took. Uses the metre stride length from way above
    running_speed = avg_overall_stride_length / stride_time if stride_time > 0 else 0

    #How much you bob up and down vs how far forward each stride takes you. Garmin/Stryd's headline efficiency metric.
    #Recomputing stride length in raw px here so it matches the oscillation units (both are normalised px)
    def stride_length_px(gait_cycle, ankle_arr):
        lengths = []
        arr = np.array(ankle_arr)
        for start, end in gait_cycle:
            s, e = frame_to_index(start), frame_to_index(end)
            xs = arr[s:e + 1, 0]
            lengths.append(max(xs) - min(xs))
        return lengths

    px_strides = stride_length_px(extracted_left_gait_cycles, left_ankle_arr) + stride_length_px(extracted_right_gait_cycles, right_ankle_arr)
    avg_stride_px = np.mean(px_strides)
    #x100 so its a readable percent instead of a tiny decimal
    vertical_ratio = (avg_total_oscillation / avg_stride_px) * 100

    #Knee flexion at touchdown. joint_angle gives 180 for a straight leg so i flip it into "how bent". A near-straight
    #knee here is the stiff/locked landing everyone warns about - but im reporting the number, not throwing an alarm
    def knee_flexion_at_contact(strike_frames, hip_arr, knee_arr, ankle_arr):
        flexions = []
        for frame_tuple in strike_frames:
            ind = frame_to_index(frame_tuple[0])
            flexions.append(180 - joint_angle(hip_arr[ind], knee_arr[ind], ankle_arr[ind]))
        return flexions

    #Foot strike angle -> continuous version of heel/mid/fore instead of just the 3 buckets. y is DOWN in pixel space
    #so (heel_y - toe_y) > 0 means the toe sits higher than the heel = heel strike. abs on the x gap so it reads the
    #same whether the runner faces left or right
    def foot_strike_angle(strike_frames, heel_arr, toe_arr):
        angles = []
        for frame_tuple in strike_frames:
            ind = frame_to_index(frame_tuple[0])
            heel, toe = heel_arr[ind], toe_arr[ind]
            angles.append(np.rad2deg(np.arctan2(heel[1] - toe[1], abs(toe[0] - heel[0]))))
        return angles

    #Knee flexion excursion = peak bend during stance minus the bend at touchdown. Basically how much the knee GIVES
    #to soak up the landing. Small excursion + straight touchdown = stiff runner not using their knee as a spring
    def knee_flexion_excursion(strike_frames, toeoff_frames, hip_arr, knee_arr, ankle_arr):
        strikes = sorted([f[0] for f in strike_frames])
        toeoffs = sorted([f[0] for f in toeoff_frames])
        excursions = []
        for s in strikes:
            #Pair each touchdown with the next toe-off of the SAME leg - thats one stance phase
            later = [t for t in toeoffs if t > s]
            if not later:
                continue
            s_ind, t_ind = frame_to_index(s), frame_to_index(later[0])
            flexions = [180 - joint_angle(hip_arr[i], knee_arr[i], ankle_arr[i]) for i in range(s_ind, t_ind + 1)]
            if not flexions:
                continue
            #flexions[0] is the touchdown value since the window starts at the strike frame
            excursions.append(max(flexions) - flexions[0])
        return excursions

    #Hip extension at toe-off = the drive angle. Trunk(shoulder)-hip-thigh(knee). Bigger = more of the leg trailing
    #behind you at push off, which is the propulsion youre looking for
    def hip_extension_at_toeoff(toeoff_frames, shoulder_arr, hip_arr, knee_arr):
        angles = []
        for frame_tuple in toeoff_frames:
            ind = frame_to_index(frame_tuple[0])
            angles.append(joint_angle(shoulder_arr[ind], hip_arr[ind], knee_arr[ind]))
        return angles

    #Average trunk lean off vertical across the whole clip. shoulder-above-hip vector vs straight up.
    #abs it because the facing direction flips the x sign - finetune later if you wanna keep it signed (fwd vs back)
    def trunk_lean():
        leans = []
        n_sh = min(len(left_shoulder_arr), len(right_shoulder_arr))
        for i in range(n_sh):
            mid_shoulder = [(left_shoulder_arr[i][0] + right_shoulder_arr[i][0]) / 2, (left_shoulder_arr[i][1] + right_shoulder_arr[i][1]) / 2]
            dx = mid_shoulder[0] - mean_hip_x_coords[i]
            #hip_y - shoulder_y (y is down, shoulders sit above the hips so this comes out positive)
            dy = mean_hip_y_coords[i] - mid_shoulder[1]
            leans.append(np.rad2deg(np.arctan2(dx, dy)))
        return np.mean(np.abs(leans))

    diagnose_knee_flexion_timing(detected_frames, strikefoot_frames_left, strikefoot_frames_right,
                                 left_hip_arr, left_knee_arr, left_ankle_arr,
                                 right_hip_arr, right_knee_arr, right_ankle_arr)

    left_knee_flexion = np.mean(knee_flexion_at_contact(strikefoot_frames_left, left_hip_arr, left_knee_arr, left_ankle_arr))
    right_knee_flexion = np.mean(knee_flexion_at_contact(strikefoot_frames_right, right_hip_arr, right_knee_arr, right_ankle_arr))
    left_foot_strike_angle = np.mean(foot_strike_angle(strikefoot_frames_left, left_heel_kpt_arr, left_toe_kpt_arr))
    right_foot_strike_angle = np.mean(foot_strike_angle(strikefoot_frames_right, right_heel_kpt_arr, right_toe_kpt_arr))
    left_knee_excursion = np.mean(knee_flexion_excursion(strikefoot_frames_left, toe_off_frames_left, left_hip_arr, left_knee_arr, left_ankle_arr))
    right_knee_excursion = np.mean(knee_flexion_excursion(strikefoot_frames_right, toe_off_frames_right, right_hip_arr, right_knee_arr, right_ankle_arr))
    left_hip_extension = np.mean(hip_extension_at_toeoff(toe_off_frames_left, left_shoulder_arr, left_hip_arr, left_knee_arr))
    right_hip_extension = np.mean(hip_extension_at_toeoff(toe_off_frames_right, right_shoulder_arr, right_hip_arr, right_knee_arr))
    avg_trunk_lean = trunk_lean()

    #--- Spring-mass model (Morin et al. 2005). The whole point: force + stiffness estimates with NO force plate,
    #straight from contact/aerial time. Peak GRF drops out in bodyweights (mass cancels) and the COM drop comes out
    #in real metres from g*t^2, so both of those are reportable even before calibration. Stiffness needs mass. ---
    GRAVITY = 9.81

    #Peak vertical ground reaction force in BODYWEIGHTS. Typical runners sit around 2-3 BW
    peak_vgrf_bw = (np.pi / 2) * (t_a / t_c + 1) if t_c > 0 else 0

    #Vertical drop of the COM during contact. abs because Morin's formula is signed downward and i just want the magnitude
    com_vertical_drop = abs(GRAVITY * t_c ** 2 * (1/8 - (t_a / t_c + 1) / (2 * np.pi))) if t_c > 0 else 0

    vertical_stiffness = None
    leg_stiffness = None
    if mass is not None and t_c > 0:
        peak_vgrf = peak_vgrf_bw * mass * GRAVITY        #back into newtons now that we have mass
        vertical_stiffness = peak_vgrf / com_vertical_drop if com_vertical_drop > 0 else None
        #Leg length ~ 53% of standing height, the classic anthropometric shortcut so i dont need another keypoint measure
        leg_length = 0.53 * height
        #How much the "leg spring" compresses over contact. Needs speed since the leg also sweeps backwards while loaded
        delta_L = leg_length - np.sqrt(leg_length ** 2 - (running_speed * t_c / 2) ** 2) + com_vertical_drop
        leg_stiffness = peak_vgrf / delta_L if delta_L > 0 else None

    #Symmetry indices. Standard SI: 0% = perfectly even, climbs as the two legs diverge. abs so it doesnt care
    #which side is bigger. Heads up - from a single side view the far leg is half hidden so treat these as low
    #confidence unless its a clean treadmill clip 
    def symmetry_index(left, right):
        denom = 0.5 * (left + right)
        return abs(left - right) / denom * 100 if denom != 0 else 0

    stride_length_symmetry = symmetry_index(avg_left_stride_length, avg_right_stride_length)
    oscillation_symmetry = symmetry_index(avg_left_side_oscillation, avg_right_side_oscillation)
    knee_flexion_symmetry = symmetry_index(left_knee_flexion, right_knee_flexion)
    foot_strike_symmetry = symmetry_index(left_foot_strike_angle, right_foot_strike_angle)

    #Bundling everything into one dict so the RAG layer just gets a flat bag of numbers to retrieve + interpret.
    #Storing the raw continuous values on purpose (not buckets) - the normative ranges + wording live in the RAG,
    #and per that Stiffler-Joachim nonlinearity note the hard cutoffs lie to you anyway
    return {
        #core (already computed above)
        "Cadence (steps/sec)": cadence,
        "Average GCT (s)": average_gct,
        "Average Flight Time (s)": average_flight_time,
        "Average Vertical Oscillation (px)": avg_total_oscillation,
        "Average Stride Length (m)": avg_overall_stride_length,
        "Left Foot Overstrides": left_foot_overstrides,
        "Right Foot Overstrides": right_foot_overstrides,
        #timing
        "Duty Factor": duty_factor,
        "Swing Time (s)": swing_time,
        "Aerial Time (s)": t_a,
        "Contact Time (s)": t_c,
        "Stride Time (s)": stride_time,
        "Running Speed (m/s)": running_speed,
        #ratios / angles
        "Vertical Ratio (%)": vertical_ratio,
        "Trunk Lean (deg)": avg_trunk_lean,
        "Left Knee Flexion at Contact (deg)": left_knee_flexion,
        "Right Knee Flexion at Contact (deg)": right_knee_flexion,
        "Left Knee Flexion Excursion (deg)": left_knee_excursion,
        "Right Knee Flexion Excursion (deg)": right_knee_excursion,
        "Left Foot Strike Angle (deg)": left_foot_strike_angle,
        "Right Foot Strike Angle (deg)": right_foot_strike_angle,
        "Left Hip Extension at Toe-Off (deg)": left_hip_extension,
        "Right Hip Extension at Toe-Off (deg)": right_hip_extension,
        #spring-mass estimates
        "Peak Vertical GRF (BW)": peak_vgrf_bw,
        "COM Vertical Drop (m)": com_vertical_drop,
        "Vertical Stiffness (N/m)": vertical_stiffness,
        "Leg Stiffness (N/m)": leg_stiffness,
        #symmetry
        "Stride Length Symmetry (%)": stride_length_symmetry,
        "Vertical Oscillation Symmetry (%)": oscillation_symmetry,
        "Knee Flexion Symmetry (%)": knee_flexion_symmetry,
        "Foot Strike Symmetry (%)": foot_strike_symmetry,
    }

res = calculate_metrics(d_frame_preds_legs_indices, u_frame_preds_legs_indices, valid_frames, cap, non_normal_kpts, 0.4, 1.78, 70)
print(res)

with open("/Users/abhinavarora/Desktop/CadenceCV/evaluate_metrics.json", "w") as file:
    json.dump(res, file, indent=4, cls=NumpyEncoder)