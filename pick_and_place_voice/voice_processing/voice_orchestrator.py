#!/usr/bin/env python3
import os
import time
import subprocess
import rclpy
from rclpy.node import Node
import pyaudio
from ament_index_python.packages import get_package_share_directory
from dotenv import load_dotenv

# wake word & STT
from voice_processing.wakeup_word import WakeupWord
from voice_processing.stt import STT

PACKAGE = "pick_and_place_voice"
PKG_PATH = get_package_share_directory(PACKAGE)

# ── 환경 변수에서 마이크 인덱스 읽기(없으면 기본 장치 사용)
_env_idx = os.environ.get("MIC_DEVICE_INDEX", "").strip()
MIC_DEVICE_INDEX = int(_env_idx) if _env_idx.isdigit() else None

# 웨이크워드용 버퍼(스트림 chunk와 독립적으로 사용 가능)
WAKE_BUFFER_SIZE = 24000

# :작은_파란색_다이아몬드: 브링업 명령어 (환경에 맞게 수정)
BRINGUP_CMD = (
    "ros2 launch dsr_bringup2 dsr_bringup2_rviz.launch.py "
    "mode:=real host:=192.168.1.100 port:=12345 model:=m0609"
)

def run_detached(cmd: str, session_name: str = "robot_bringup"):
    """tmux로 브링업 실행, 없으면 nohup 사용"""
    try:
        # 일부 환경에서 'new'가 alias가 아닐 수 있어 'new-session'이 더 호환됨
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name, cmd], check=True)
    except Exception:
        subprocess.Popen(cmd, shell=True)

# ─────────────────────────────────────────────────────────────────────────────
# 오디오 유틸
# ─────────────────────────────────────────────────────────────────────────────
def best_chunk(rate: int) -> int:
    """약 10ms 버퍼(지연↓, 안정↑). 너무 작으면 끊김, 너무 크면 딜레이 큼."""
    return max(256, rate // 100)

def open_input_stream(pa: pyaudio.PyAudio, device_index: int | None):
    """
    장치가 지원하는 (channels, rate, chunk) 조합을 자동으로 탐색하여 스트림을 연다.
    우선순위: 사용자가 지정한 채널/레이트 → 48k → 44.1k → 32k → 16k / 채널은 1ch→2ch 순
    """
    # 장치 정보 가져오기(인덱스가 없거나 잘못되면 기본 입력 장치 사용)
    if device_index is not None:
        try:
            dev_info = pa.get_device_info_by_index(device_index)
        except Exception:
            dev_info = pa.get_default_input_device_info()
            device_index = int(dev_info["index"])
    else:
        dev_info = pa.get_default_input_device_info()
        device_index = int(dev_info["index"])

    max_in = int(dev_info.get("maxInputChannels", 0))
    if max_in <= 0:
        raise RuntimeError(f"입력 채널을 지원하지 않는 장치입니다: {dev_info.get('name')}")

    # 채널 후보: 1ch 우선, 안 되면 2ch (장치 max 범위 내)
    ch_candidates = [c for c in (1, 2) if c <= max_in]

    # 샘플레이트 후보
    rate_candidates = [48000, 44100, 32000, 16000]

    last_err = None
    for ch in ch_candidates:
        for rate in rate_candidates:
            try:
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=ch,
                    rate=rate,
                    input=True,
                    frames_per_buffer=best_chunk(rate),
                    input_device_index=device_index
                )
                # 성공: 스트림, 실제 파라미터 반환
                return stream, device_index, ch, rate, best_chunk(rate)
            except Exception as e:
                last_err = e
                continue
    raise OSError(f"오디오 스트림을 열 수 없습니다: {last_err}")

# ─────────────────────────────────────────────────────────────────────────────
# 메인 노드
# ─────────────────────────────────────────────────────────────────────────────
class VoiceBringup(Node):
    def __init__(self):
        super().__init__("voice_bringup_node")

        # --- 마이크 & 웨이크워드 ---
        self.pa = pyaudio.PyAudio()
        try:
            self.stream, self.device_index, self.channels, self.rate, self.chunk = open_input_stream(
                self.pa, MIC_DEVICE_INDEX
            )
            self.get_logger().info(
                f"Audio opened ▶ device={self.device_index}, ch={self.channels}, rate={self.rate}, chunk={self.chunk}"
            )
        except Exception as e:
            self.get_logger().error(f"마이크 열기 실패: {e}")
            # 노드 초기화 실패로 종료
            raise

        # 웨이크워드 모듈 초기화(내부 버퍼는 스트림 chunk와 별도)
        self.wake = WakeupWord(buffer_size=WAKE_BUFFER_SIZE)
        self.wake.set_stream(self.stream)

        # --- STT ---
        load_dotenv(os.path.join(PKG_PATH, "resource/.env"))
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.stt = STT(openai_api_key=openai_api_key)
        self.stt.duration = 10  # 브링업 후 10초간 대기

        self.armed = False
        self.timer = self.create_timer(0.05, self.loop)

    def loop(self):
        # 웨이크워드 감지
        if not self.armed:
            if not self.wake.is_wakeup():
                return
            self.get_logger().warn("[Wake] Hello Rokey detected → bringup 실행")
            run_detached(BRINGUP_CMD)

            # 브링업 대기 (필요시 토픽 확인 로직으로 교체 가능)

            self.armed = True
            self.get_logger().info("브링업 완료. 이제 10초간 음성 인식 대기")

            # 10초 STT
            try:
                text = self.stt.speech2text()
                self.get_logger().info(f"[STT 결과] {text}")
            except Exception as e:
                self.get_logger().error(f"STT 오류: {e}")

            # 완료 후 다시 대기 상태
            self.armed = False

    def destroy_node(self):
        # 자원 정리
        try:
            if hasattr(self, "stream") and self.stream is not None:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
        except Exception:
            pass
        try:
            if hasattr(self, "pa") and self.pa is not None:
                self.pa.terminate()
        except Exception:
            pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VoiceBringup()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
