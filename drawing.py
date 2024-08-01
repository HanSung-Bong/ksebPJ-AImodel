import cv2
import os

# 비디오 설정
output_video_path = '/content/gdrive/MyDrive/KSEB_익명성/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30.0
frame_size = (1920, 1080)
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

for i in range(1, 751):
    input_image_path = f'/content/gdrive/MyDrive/KSEB_익명성/0_dataset/SNGS-031/img1/{i:06d}.jpg'
    canvas = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    j = i - 1
    file_path = f"/content/gdrive/MyDrive/KSEB_익명성/envs_test/txt/{j:03d}.txt"
    
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 선수 정보 처리 및 그리기
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 8:
            class_id, x, y, width, height, player_id, team_id, jersey_number = map(int, parts)

            # 호의 중심점과 크기 계산
            center = (x, y + height // 2 + 3)  # y좌표 아래쪽에 위치하도록 조정
            axes = (width // 2, height // 4)
            angle = 0
            start_angle = 0
            end_angle = 180
            arc_color = (255, 0, 0) if team_id == 0 else (0, 0, 255)  # 팀에 따라 색상 변경
            thickness = 2
            cv2.ellipse(canvas, center, axes, angle, start_angle, end_angle, arc_color, thickness)

            # 등번호 쓰기
            text = str(jersey_number)
            org = (x - 10, y + height // 2 + 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            text_color = (0, 0, 0)
            text_thickness = 2
            cv2.putText(canvas, text, org, font, font_scale, text_color, text_thickness)

    # 비디오에 프레임 추가
    out.write(canvas)

# 비디오 릴리즈
out.release()

