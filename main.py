import sys

from scrfd_ import SCRFD
from scrfd_deepsort import *

sys.path.append('/home/tima/detec_and_tracking/Face-Mask-Detection')
from detect_mask_image import detect_mask_image, load_model_detect_mask

model_mask =load_model_detect_mask()
detect_mask = detect_mask_image(model=model_mask)


detector = SCRFD(model_file='/home/tima/detec_and_tracking/insightface/detection/scrfd/onnx/scrfd_500m.onnx')
detector.prepare(-1)

#http://172.16.4.22:4747/video
# tracker = SCRFD_DeepSort(model_tracking_path="/home/tima/detec_and_tracking/deep_sort/deep_sort/model_weights/mars-small128.pb", detector=detector, model_mask=model_mask, detect = detect_mask)

# tracker.track_video("5.mp4", output=None, show_live = True, skip_frames = 0, count_objects = True, verbose=1)

import cv2
import threading

# Hàm để xử lý một video
def process_video(video_path, window_name):
    # Khởi tạo và cấu hình đối tượng SCRFD_DeepSort
    tracker = SCRFD_DeepSort(model_tracking_path="/home/tima/detec_and_tracking/deep_sort/deep_sort/model_weights/mars-small128.pb", detector=detector, model_mask=model_mask, detect = detect_mask)

    tracker.track_video(video_path, output=None, show_live = True, skip_frames = 0, count_objects = True, verbose=1)

    
# Danh sách các video cần xử lý
video_paths = ['5.mp4']

# Tạo một luồng riêng và cửa sổ hiển thị cho mỗi video
threads = []
for i, video_path in enumerate(video_paths):
    window_name = f"Video {i+1}"
    thread = threading.Thread(target=process_video, args=(video_path, window_name))
    threads.append(thread)
    thread.start()

# Hiển thị video từ các luồng
for i, thread in enumerate(threads):
    window_name = f"Video {i+1}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    # Tạo cửa sổ hiển thị cho mỗi luồng
    for i, thread in enumerate(threads):
        window_name = f"Video {i+1}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        for i, thread in enumerate(threads):
            window_name = f"Video {i+1}"
            if not thread.is_alive():
                cv2.destroyWindow(window_name)
                threads.remove(thread)
        
        if len(threads) == 0:
            break

        # Hiển thị frame từ mỗi luồng
        for i, thread in enumerate(threads):
            window_name = f"Video {i+1}"
            frame = thread.tracker.get_frame()
            cv2.imshow(window_name, frame)

        # Thoát khỏi vòng lặp nếu người dùng nhấn phím 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng bộ nhớ và đóng cửa sổ hiển thị
    cv2.destroyAllWindows()


# Giải phóng bộ nhớ và đóng cửa sổ hiển thị
cv2.destroyAllWindows()

