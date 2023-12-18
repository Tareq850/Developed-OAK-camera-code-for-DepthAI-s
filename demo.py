#!/usr/bin/env python3
import cv2
from math import atan2, degrees
import argparse
from BlazeposeRenderer import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT

# For gesture demo
semaphore_flag = {
    (3, 4): 'A', (2, 4): 'B', (1, 4): 'C', (0, 4): 'D',
    (4, 7): 'E', (4, 6): 'F', (4, 5): 'G', (2, 3): 'H',
    (0, 3): 'I', (0, 6): 'J', (3, 0): 'K', (3, 7): 'L',
    (3, 6): 'M', (3, 5): 'N', (2, 1): 'O', (2, 0): 'P',
    (2, 7): 'Q', (2, 6): 'R', (2, 5): 'S', (1, 0): 'T',
    (1, 7): 'U', (0, 5): 'V', (7, 6): 'W', (7, 5): 'X',
    (1, 6): 'Y', (5, 6): 'Z',
}
def angle_with_y(v):
    # v: 2d vector (x,y)
    # Returns angle in degree of v with y-axis of image plane
    if v[1] == 0:
        return 90
    angle = atan2(v[0], v[1])
    return degrees(angle)
def recognize_gesture(b):  
    # b: body         

    # For the demo, we want to recognize the flag semaphore alphabet
    # For this task, we just need to measure the angles of both arms with vertical
    right_arm_angle = angle_with_y(b.landmarks[KEYPOINT_DICT['right_elbow'], :2] - b.landmarks[KEYPOINT_DICT['right_shoulder'], :2])
    left_arm_angle = angle_with_y(b.landmarks[KEYPOINT_DICT['left_elbow'], :2] - b.landmarks[KEYPOINT_DICT['left_shoulder'], :2])
    right_pose = int((right_arm_angle + 202.5) / 45) % 8 
    left_pose = int((left_arm_angle + 202.5) / 45) % 8
    letter = semaphore_flag.get((right_pose, left_pose), None)
    return letter
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--edge', action="store_true",
                    help="Use Edge mode (postprocessing runs on the device)")
parser_tracker = parser.add_argument_group("Tracker arguments")
parser_tracker.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='full',
                    help="Landmark model to use (default=%(default)s")
parser_tracker.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")                   
parser_tracker.add_argument("--pd_m", type=str,
                    help="Path to an .blob file for pose detection model")
parser_tracker.add_argument("--lm_m", type=str,
                    help="Landmark model ('full' or 'lite' or 'heavy') or path to an .blob file")
parser_tracker.add_argument('-xyz', '--xyz', action="store_true", 
                    help="Get (x,y,z) coords of reference body keypoint in camera coord system (only for compatible devices)")
parser_tracker.add_argument('-c', '--crop', action="store_true", 
                    help="Center crop frames to a square shape before feeding pose detection model")
parser_tracker.add_argument('--no_smoothing', action="store_true", 
                    help="Disable smoothing filter")
parser_tracker.add_argument('-f', '--internal_fps', type=int, 
                    help="Fps of internal color camera. Too high value lower NN fps (default= depends on the model)")                    
parser_tracker.add_argument('--internal_frame_height', type=int, default=640,                                                                                    
                    help="Internal color camera frame height in pixels (default=%(default)i)")                    
parser_tracker.add_argument('-s', '--stats', action="store_true", 
                    help="Print some statistics at exit")
parser_tracker.add_argument('-t', '--trace', action="store_true", 
                    help="Print some debug messages")
parser_tracker.add_argument('--force_detection', action="store_true", 
                    help="Force person detection on every frame (never use landmarks from previous frame to determine ROI)")
parser_renderer = parser.add_argument_group("Renderer arguments")
parser_renderer.add_argument('-3', '--show_3d', choices=[None, "image", "world", "mixed"], default=None,
                    help="Display skeleton in 3d in a separate window. See README for description.")
parser_renderer.add_argument("-o","--output",
                    help="Path to output video file")
args = parser.parse_args()
if args.edge:
    from BlazeposeDepthaiEdge import BlazeposeDepthai
else:
    from BlazeposeDepthai import BlazeposeDepthai
tracker = BlazeposeDepthai(input_src=args.input, 
            pd_model=args.pd_m,
            lm_model=args.lm_m,
            smoothing=not args.no_smoothing,   
            xyz=args.xyz,            
            crop=args.crop,
            internal_fps=args.internal_fps,
            internal_frame_height=args.internal_frame_height,
            force_detection=args.force_detection,
            stats=True,
            trace=args.trace)   
renderer = BlazeposeRenderer(
                tracker, 
                show_3d=args.show_3d, 
                output=args.output)
KEYPOINT_DICT_INV = {v: k for k, v in KEYPOINT_DICT.items()}
while True:
    # Run blazepose on the next frame
    frame, body = tracker.next_frame()
    if frame is None: 
        break
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    if body:
        angles_values = []
        keypoint_pairs = [
            (11, 13, 15),
            (16, 14, 12),
            (11, 23, 25),
            (23, 25, 27),]
        for keypoint1, keypoint2, keypoint3 in keypoint_pairs:
            point1_name = KEYPOINT_DICT_INV[keypoint1]
            point2_name = KEYPOINT_DICT_INV[keypoint2]
            point3_name = KEYPOINT_DICT_INV[keypoint3]
            angle = angle_with_y(body.landmarks[keypoint3, :2] - body.landmarks[keypoint2, :2])
            angles_values.append((f"{keypoint1}-{keypoint2}-{keypoint3} ({point1_name}-{point2_name}-{point3_name})", angle)) 
        cv2.putText(frame, "", (frame.shape[1] - 420, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        y_position = 50
        for key, value in angles_values:
            cv2.putText(frame, f"{key} Angle: {value:.2f}", (frame.shape[1] - 420, y_position), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            y_position += 20
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
renderer.exit()
tracker.exit()