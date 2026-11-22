# 24/365 모델 후속 실험

2025년 24/365는 영상에서 화염과 연기를 찾아 조기 감지에 활용하기 위한 팀 프로젝트였다. 당시 프로젝트는 친구의 GitHub 저장소에 최종본이 남아 있으며, 로컬 `24365-MASAN-STRRET-main` 폴더가 그 스냅샷이다. 이 저장소는 2026년 후속 모델 실험을 기록한다. [포트폴리오 계획](PORTFOLIO_PLAN.md)

## 구조

| 경로 | 역할 |
|---|---|
| `src/wildfire_detection/check_data.py` | YOLO 라벨 검증, 분할별 집계, 이미지·라벨 SHA-256 |
| `src/wildfire_detection/train.py` | 버전 관리된 설정으로 학습하고 가중치 저장 |
| `src/wildfire_detection/evaluate.py` | Ultralytics AP와 고정 confidence에서의 TP·FP·FN, 오류 예시 |
| `src/wildfire_detection/run_records.py` | 데이터 지문, 환경, 커밋을 실행에 연결 |
| `configs/` | 데이터 위치 예시와 YOLO11s 학습 설정 |
| `pyproject.toml` | 패키지 설치 설정과 의존성 |
| `tests/` | 데이터 분할과 탐지 매칭의 회귀 테스트 |
| `runs/` | 로컬 실행 산출물 |

학습과 평가는 Terminal에서 실행한다. 분석 노트북은 실제 결과를 확보한 뒤 저장된 지표와 예시 이미지를 읽는 형태로 추가할 수 있다.

## 환경

저장소 루트에서 Bash 또는 Zsh로 실행한다. Python 3.12를 사용한다. 학교의 NVIDIA 워크스테이션에서는 먼저 `nvidia-smi`를 확인한 뒤, 가상환경에 장비와 드라이버에 맞는 PyTorch와 torchvision을 설치한다. 맥북에서 시험 실행할 때는 `--device cpu`를 사용한다. [PyTorch 설치 안내](https://docs.pytorch.org/get-started/locally/)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# 학교 워크스테이션에서는 여기서 GPU용 PyTorch와 torchvision을 먼저 설치한다.
python -m pip install -e .
python -m pip check
python -m unittest discover -s tests -v
```

워크스테이션에서 GPU를 쓸 때는 다음 출력에 GPU 이름이 나오는지 확인한다.

```bash
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

## 1. 데이터 확인

Roboflow v2를 YOLO 형식으로 받아 `train/images`, `valid/images`, `test/images` 및 각 `labels` 폴더가 있는 위치에 압축을 푼다. 아래 명령 뒤 **로컬 `data.yaml`**의 `path`와 분할 경로를 수정한다. `names`의 숫자와 클래스 매핑은 예시 파일의 순서를 그대로 쓰기 전에 실제 export와 대조한다. [Roboflow v2](https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia/dataset/2)

```bash
cp configs/data.example.yaml data.yaml
# data.yaml을 편집한 뒤 실행
python -m wildfire_detection.check_data --data data.yaml
```

이미지와 라벨의 대응, 클래스 ID, 좌표, 분할별 이미지·객체 수, **바이트가 같은 이미지의 분할 간 중복**을 확인한다. 배경 이미지로 확인한 경우 대응 위치에 내용이 없는 라벨 파일을 둔다. 화면이 비슷하거나 같은 영상에서 나온 재인코딩 프레임은 별도로 확인해야 한다. 라벨이나 분할을 바꿀 때는 새 데이터 버전과 새 폴더를 사용한다. Ultralytics가 만든 라벨 캐시까지 기존 경로에 남아 있으면 변경 전 파일을 읽을 수 있다.

## 2. 학습

학교 GPU에서 3 epochs로 데이터 로딩과 메모리 사용량, 라벨 시각화, epoch 시간을 확인한다. 아래 장비 번호 `0`은 현재 환경에서 첫 번째로 보이는 CUDA GPU다.

```bash
python -m wildfire_detection.train --data data.yaml --dataset-version roboflow-v2 \
  --device 0 --epochs 3 --name pilot-yolo11s
```

반복 실행에는 새 이름을 사용한다. 시험 실행에서 클래스 매핑과 메모리 사용량을 확인한 뒤 본 학습을 진행한다. 기본 설정은 `configs/yolo11s.yaml`의 50 epochs, batch 8, 이미지 크기 640이다. 설정을 변경하면 별도 YAML로 저장하고 `--config`에 지정한다. YOLO11s 사전학습 가중치는 첫 실행 시 다운로드된다. 가중치를 미리 준비했다면 `--model`에 파일 경로를 지정한다.

```bash
python -m wildfire_detection.train --data data.yaml --dataset-version roboflow-v2 \
  --device 0 --config configs/yolo11s.yaml --name yolo11s-baseline
```

## 3. 평가

validation split에서 운영 confidence를 고른다. 예시 명령의 `0.25`는 비교 출발점이다. 선정 후 test split에서 평가하고, 두 분할의 해석을 구분해 기록한다.

```bash
python -m wildfire_detection.evaluate --data data.yaml --dataset-version roboflow-v2 \
  --weights runs/yolo11s-baseline/train/weights/best.pt \
  --device 0 --split val --confidence 0.25 --name yolo11s-val-c025

# 아래 값은 validation에서 정한 confidence로 교체한다.
CONFIDENCE=0.25
python -m wildfire_detection.evaluate --data data.yaml --dataset-version roboflow-v2 \
  --weights runs/yolo11s-baseline/train/weights/best.pt \
  --device 0 --split test --confidence "$CONFIDENCE" --name yolo11s-test
```

- **AP**는 Ultralytics 평가기로 계산한다. `metrics.json`에 전체 mAP와 클래스별 AP, `ap/`에 평가 그림이 저장된다. 해당 분할에 정답이 없는 클래스의 AP는 `null`이다. 학습 이미지 크기를 바꿨다면 평가의 `--imgsz`에도 그 값을 지정한다. [Ultralytics 평가 문서](https://docs.ultralytics.com/modes/val/)
- **고정 confidence 지표**는 해당 임계값으로 다시 추론해 계산한다. 높은 confidence의 예측부터 같은 클래스의 아직 매칭되지 않은 정답 중 최대 IoU가 0.50 이상인 박스에 일대일로 연결한다. 분모가 0인 지표는 `null`이다.
- `errors.jsonl`에 모든 FP·FN 좌표가 들어 있다. `examples/`에는 경로 순서상 첫 오류 이미지가 기본 12장까지 저장된다. 이 이미지들은 검토 출발점이다. 사례를 고를 때는 오류 기록과 원본 라벨을 함께 확인한다.

## 실행 증거 보관

각 실행에 `run.json`의 커밋 SHA, 명령, 장비와 가중치 해시, `dataset.json`의 분할 및 데이터 지문, `environment.txt`의 패키지 버전이 저장된다. `train/`에는 `args.yaml`, `results.csv`, `weights/best.pt`가 생성된다. 실행 폴더를 보관하면 결과 표를 만든 조건과 가중치를 다시 확인할 수 있다.

데이터, 학습 결과, 로컬 설정, 가중치는 Git 제외 대상이다. 학교에서 허용한 장소에 실행 산출물을 백업하고, 공개할 때는 원본 이미지의 이용 권한과 로컬 절대 경로의 노출 여부를 확인한다.
