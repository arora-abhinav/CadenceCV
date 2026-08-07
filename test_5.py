# Colour-based leg tracking. The runner wears a different coloured sock on each leg (colours from opposite
# ends of the spectrum so they never get confused). Each leg's identity comes straight from its sock colour,
# so there is no geometry, no motion analysis, and no anchor. We just find each colour's blob per frame and
# that blob IS that leg. The user states which colour is on which leg via the two config constants below.
import json
import cv2
import numpy as np

VIDEO_PATH = "/Users/abhinavarora/Desktop/CadenceCV/Videos/colour_socks_test.mp4"
POSITIONS_OUT = "/Users/abhinavarora/Desktop/CadenceCV/sock_positions.json"

# --- the user answers "which sock is on which leg" here ---
LEFT_LEG_COLOUR = "green"
RIGHT_LEG_COLOUR = "magenta"

# a blob smaller than this many pixels is treated as noise, not a sock
MINIMUM_BLOB_AREA = 150

# display gets shrunk if the frame is taller than this
MAXIMUM_DISPLAY_HEIGHT = 900

# HSV colour ranges (OpenCV HSV: hue 0-179, sat 0-255, val 0-255). Each colour maps to a list of ranges,
# because red wraps around the hue boundary and needs two. These are starting points - tune them to the
# actual socks under the actual lighting.
COLOUR_RANGES = {
    "red": [
        ((0, 120, 70), (10, 255, 255)),
        ((170, 120, 70), (180, 255, 255)),
    ],
    "green": [
        ((40, 70, 70), (80, 255, 255)),
    ],
    "blue": [
        ((100, 120, 70), (130, 255, 255)),
    ],
    "yellow": [
        ((20, 100, 100), (35, 255, 255)),
    ],
    "orange": [
        ((10, 120, 120), (20, 255, 255)),
    ],
    "cyan": [
        ((85, 100, 100), (100, 255, 255)),
    ],
    "magenta": [
        ((140, 80, 80), (170, 255, 255)),
    ],
}

# BGR colours used only to draw each leg's marker on screen
LEFT_MARKER_BGR = (255, 0, 0)
RIGHT_MARKER_BGR = (0, 140, 255)


def build_colour_mask(hsv_frame, colour_name):
    # combine every HSV range that belongs to this colour into one binary mask
    ranges_for_this_colour = COLOUR_RANGES[colour_name]
    combined_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for lower_bound, upper_bound in ranges_for_this_colour:
        lower_array = np.array(lower_bound, dtype=np.uint8)
        upper_array = np.array(upper_bound, dtype=np.uint8)
        single_range_mask = cv2.inRange(hsv_frame, lower_array, upper_array)
        combined_mask = cv2.bitwise_or(combined_mask, single_range_mask)
    return combined_mask


def clean_mask(raw_mask):
    # open then close to drop speckle noise and fill small holes in the sock blob
    kernel = np.ones((5, 5), dtype=np.uint8)
    opened_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    return closed_mask


def find_largest_blob_centroid(mask):
    # return the centroid of the biggest blob in the mask, or None if there isn't a big-enough one
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour_area = cv2.contourArea(largest_contour)
    if largest_contour_area < MINIMUM_BLOB_AREA:
        return None
    moments = cv2.moments(largest_contour)
    if moments["m00"] == 0:
        return None
    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])
    return (centroid_x, centroid_y)


def locate_sock(hsv_frame, colour_name):
    # full pipeline for one sock: mask the colour, clean it, return its centroid (or None)
    raw_mask = build_colour_mask(hsv_frame, colour_name)
    cleaned_mask = clean_mask(raw_mask)
    centroid = find_largest_blob_centroid(cleaned_mask)
    return centroid


def draw_leg_marker(frame, position, leg_label, marker_bgr):
    # draw a filled dot + a leg label at the sock position
    if position is None:
        return
    marker_x = position[0]
    marker_y = position[1]
    cv2.circle(frame, (marker_x, marker_y), 10, marker_bgr, -1)
    cv2.putText(frame, leg_label, (marker_x + 12, marker_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, marker_bgr, 2)


def main():
    print("left leg  =", LEFT_LEG_COLOUR, "sock")
    print("right leg =", RIGHT_LEG_COLOUR, "sock")

    video_capture = cv2.VideoCapture(VIDEO_PATH)

    # one record per frame: the pixel position of each leg's sock (or None when not visible)
    per_frame_positions = []

    frame_index = 0
    while True:
        frame_was_read, frame = video_capture.read()
        if not frame_was_read:
            break

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        left_sock_position = locate_sock(hsv_frame, LEFT_LEG_COLOUR)
        right_sock_position = locate_sock(hsv_frame, RIGHT_LEG_COLOUR)

        per_frame_positions.append({
            "frame": frame_index,
            "left_leg": left_sock_position,
            "right_leg": right_sock_position,
        })

        draw_leg_marker(frame, left_sock_position, "LEFT", LEFT_MARKER_BGR)
        draw_leg_marker(frame, right_sock_position, "RIGHT", RIGHT_MARKER_BGR)

        banner_text = "frame " + str(frame_index) + "   space/n next   q quit"
        cv2.putText(frame, banner_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        if frame_height > MAXIMUM_DISPLAY_HEIGHT:
            display_scale = MAXIMUM_DISPLAY_HEIGHT / frame_height
            display_width = int(frame_width * display_scale)
            frame = cv2.resize(frame, (display_width, MAXIMUM_DISPLAY_HEIGHT))

        cv2.imshow("coloured sock leg tracking", frame)

        pressed_key = cv2.waitKey(0) & 0xFF
        if pressed_key == ord("q") or pressed_key == 27:
            break

        frame_index = frame_index + 1

    video_capture.release()
    cv2.destroyAllWindows()

    output_file = open(POSITIONS_OUT, "w")
    json.dump(per_frame_positions, output_file, indent=2)
    output_file.close()
    print("saved", len(per_frame_positions), "frames of sock positions ->", POSITIONS_OUT)


if __name__ == "__main__":
    main()
