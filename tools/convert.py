import os
import cv2
import argparse
from PIL import Image



def mp4_to_gif(mp4_path: str, gif_path: str, frame_skip: int = 1, resize_factor: float = 1.0):
    """
    MP4 파일을 읽어 움직이는 GIF로 저장합니다.

    Args:
        mp4_path (str): 입력 MP4 파일 경로.
        gif_path (str): 출력 GIF 파일 경로.
        frame_skip (int): 프레임 건너뛰기 간격 (1은 모든 프레임 사용).
        resize_factor (float): GIF 크기를 조정하는 비율 (1.0은 원본 크기).

    # 사용 예시
    mp4_to_gif("input.mp4", "output.gif", frame_skip=2, resize_factor=0.5)
    """
    # MP4 파일 열기
    video = cv2.VideoCapture(mp4_path)
    if not video.isOpened():
        raise ValueError(f"Cannot open the video file: {mp4_path}")
    gif_duration = int(1000 / video.get(cv2.CAP_PROP_FPS))



    frames = []
    frame_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 프레임 건너뛰기
        if frame_count % frame_skip == 0:
            # BGR -> RGB 변환
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 크기 조정
            if resize_factor != 1.0:
                width = int(frame.shape[1] * resize_factor)
                height = int(frame.shape[0] * resize_factor)
                frame_rgb = cv2.resize(frame_rgb, (width, height))

            # 프레임을 Pillow 이미지로 변환
            frames.append(Image.fromarray(frame_rgb))

        frame_count += 1



    video.release()

    # GIF로 저장
    if len(frames) > 0:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=gif_duration  # 프레임 속도 유지
        )
        print(f"GIF saved to {gif_path}")
    else:
        raise ValueError("No frames extracted from the video.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    # eval
    parser.add_argument("--mp4_path", default=None, help="full path with mp4 file", type=str, required=True)
    parser.add_argument("--gif_filename", type=str, default=None, help="gif filename")
    parser.add_argument("--frame_skip", type=int, default=1, help="frame skip")
    parser.add_argument("--resize_factor", type=float, default=0.25, help="save resized factor")
    args = parser.parse_args()

    # gif_filename 설정
    gif_filename = os.path.splitext(args.mp4_path)[0] if args.gif_filename is None else args.gif_filename

    # gif_path 설정
    gif_path = os.path.join(os.path.dirname(args.mp4_path), f"{gif_filename}.gif")

    print(f"GIF 파일 경로: {gif_path}")
    mp4_to_gif(mp4_path=args.mp4_path, gif_path=gif_path, resize_factor=args.resize_factor)



