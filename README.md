# 24/365 모델 후속 실험

2025년 24/365 팀 프로젝트에서 다룬 화염·연기 객체 탐지 문제를 이어서 검증하는 독립 저장소다. YOLO11s를 학습하고, 고정한 테스트 데이터에서 클래스별 성능과 실패 사례를 기록한다. 실험은 준비 단계이며 후속 성능값은 아직 없다.

[포트폴리오 계획](PORTFOLIO_PLAN.md)에 원래 팀 프로젝트의 증거, 실험 범위, Featured 공개 기준을 정리했다.

## 맥북에서 준비

이 디렉터리를 새 GitHub 저장소의 루트로 사용한다. Python 3.12를 기준으로 준비하고, 기존 팀 프로젝트 폴더나 인증 파일을 복사하지 않는다.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
yolo version
```

Ultralytics는 기존 체크포인트에 기록된 버전인 `8.3.220`으로 고정했다. PyTorch 빌드는 장비에 맞춰 선택하고, 실제 학습 시 설치된 버전 전체를 실행 기록에 남긴다. [Ultralytics 8.3.220](https://pypi.org/project/ultralytics/8.3.220/)

첫 커밋 전에는 이 디렉터리에서 `git status --short --untracked-files=all`로 공개 대상 파일을 확인한다. 첫 커밋을 만든 뒤 학교 워크스테이션에서 접근 가능한 GitHub 저장소에 올린다.

## 학교 워크스테이션에서 준비

1. 이 저장소를 클론하고 Python 3.12 가상환경을 만든다. 아래 예시는 Bash/Zsh 기준이다.
2. `nvidia-smi`로 GPU와 드라이버를 확인한다. 가상환경에서 OS와 GPU에 맞는 PyTorch 및 torchvision을 먼저 설치한 뒤 프로젝트 의존성을 설치한다. [PyTorch 설치 안내](https://docs.pytorch.org/get-started/locally/)
3. Roboflow v2를 YOLO 형식으로 내보낸 뒤, export의 `data.yaml`을 확인해 로컬 설정을 만든다. `data.example.yaml`은 디렉터리 구조 예시다. `path`를 실제 절대 경로로 바꾸고 `train`, `val`, `test`와 `names`의 ID 순서를 export와 대조한다. 압축을 풀거나 폴더를 옮겼다면 각 경로에 이미지와 대응 라벨이 있는지 확인한다. [Roboflow v2](https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia/dataset/2)

```bash
nvidia-smi
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
cp data.example.yaml data.yaml
```

`data.yaml`은 수정 후 사용한다. 데이터, 로컬 경로 설정, 인증 정보, 가중치와 실행 결과는 Git에서 제외한다.

## 짧은 실행으로 확인

```bash
yolo detect train model=yolo11s.pt data=data.yaml device=0 \
  imgsz=640 batch=8 epochs=3 seed=0 project=runs name=smoke-test
```

완료 기준은 GPU에서 학습과 검증이 진행되고, 학습 배치 이미지의 클래스·박스가 올바르며 `results.csv`와 가중치가 생성되는 것이다. 두 번째·세 번째 epoch의 시간과 최대 메모리 사용량을 보고 본 학습 예산을 정한다. `batch=8`은 시작값이다.

## 본 학습과 테스트

아래는 **50 epochs로 시작하는 예시**다. 짧은 실행을 보고 batch와 epoch 예산을 결정한다. 새 학습을 시작할 때는 깨끗한 커밋을 사용하고 실제 실행 명령을 보관한다.

```bash
RUN_ID="yolo11s-$(date +%Y%m%d-%H%M%S)"
mkdir -p runs
set -o pipefail
yolo detect train model=yolo11s.pt data=data.yaml device=0 \
  imgsz=640 batch=8 epochs=50 seed=0 project=runs name="$RUN_ID" \
  2>&1 | tee "runs/$RUN_ID.train.log"

git rev-parse HEAD > "runs/$RUN_ID/git-commit.txt"
python -m pip freeze > "runs/$RUN_ID/environment.txt"
nvidia-smi > "runs/$RUN_ID/gpu.txt"
```

validation 데이터에서 모델과 운영 confidence를 고르고 기록한 뒤, 고정한 test split을 최종 평가한다. 아래 명령은 **AP 집계용**이며 운영 confidence에서의 재현율은 선택한 임계값을 명시해 별도로 계산한다. 학습을 끝낸 디렉터리 외에서 실행한다면 `RUN_ID`를 실제 학습 폴더 이름으로 설정한다. [Ultralytics 평가 문서](https://docs.ultralytics.com/modes/val/)

```bash
yolo detect val model="runs/$RUN_ID/weights/best.pt" data=data.yaml \
  split=test device=0 imgsz=640 batch=8 verbose=True plots=True \
  project="runs/$RUN_ID" name=test \
  2>&1 | tee "runs/$RUN_ID/test.log"
```

실험이 끝나면 학습 로그, 실행 폴더, 사용한 데이터 버전·분할 목록, `data.yaml` 사본과 평가 결과를 학교에서 허용된 저장소에 백업한다. 포트폴리오에 인용할 때는 가중치와 데이터의 SHA-256 또는 변경되지 않는 버전 식별자도 기록한다.
