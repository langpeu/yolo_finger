# 수어 학습기 by Langpeu — 작업 안내

웹캠으로 한국 수어 단어를 녹화·학습·실시간 인식하는 윈도우용 프로그램.
모든 조작은 로컬 웹페이지(http://127.0.0.1:8000)에서 한다.
저장소: https://github.com/langpeu/yolo_finger (branch `main`, AGPL-3.0, © 2026 Langpeu)

## 새 세션에서 바로 이어서 하기

처음 열면 이 파일을 다 읽고, 아래 순서로 상태를 확인한 뒤 "남은 일"부터 이어서 한다.

1. **작업 폴더**: 사용자 PC의 `C:\yolo_finger` 가 git 저장소다 (2026-11-05부터 여기서 작업). PC 이름 DESKTOP-US66UJG.
   - device_bash에서는 `"$HOME/mnt/C:--yolo_finger"` 로 보인다. 안 보이면 device_request_folder_access 로 `C:\yolo_finger` 를 요청한다.
   - git이 잠금 파일을 지워야 하므로 device_request_delete_permission 으로 `C:\yolo_finger` 삭제 권한을 받는다. `data/` 는 지우지 않는다.
   - 시작할 때 `git pull --ff-only` (사용자가 GitHub에서 직접 고칠 때가 있다).
   - 저장소 설정(.git/config): `core.autocrlf=true`, `user.name=Langpeu`, `user.email=8281485+langpeu@users.noreply.github.com`.
   - 연결 도구(remote-devices)가 없으면 PC와 연결이 안 된 것이다. 데스크톱 앱에서 이 대화를 열어 연결하라고 안내한다.
   - `data/` 안의 사용자 녹화·모델은 읽거나 가져오지 않는다 (아래 "사용자 데이터").
2. **고치기**: PC 파일을 device_bash 에서 직접 고친다 (python으로 읽고-바꾸고-쓰기, sed -i).
3. **테스트**: 사용자 데이터 없이 한다. 클라우드 작업 공간에 `git clone https://github.com/langpeu/yolo_finger` 해서
   ```
   pip install -r requirements.txt   # 처음 한 번 (torch는 cpu판이면 충분)
   SIGN_FAKE_CAM=1 SIGN_NO_BROWSER=1 SIGN_PORT=8775 python3 app/web_app.py
   ```
   고친 파일을 클론에 옮겨 Playwright로 확인한다. 테스트 뒤 클론에 생긴 `data/` 는 지운다.
   실제 웹캠·GPU 확인은 사용자가 `1_web.bat` 으로 해 보고 알려 준다.
4. **커밋과 push**: PC에서 `git commit` (Co-Authored-By·Claude-Session 같은 Claude 표시 줄은 붙이지 않는다, 사용자 요청).
   PC에는 GitHub 로그인 정보가 없어 push가 안 된다. 그래서
   - PC: `git bundle create "$HOME/push.bundle" origin/main..main` (device_bash 작업 폴더 쪽, 저장소 밖)
   - device_stage_files 로 묶음 파일을 클라우드로 가져와 클론에서 `git pull <묶음> main` 후 `git push origin main`
   - PC: `git fetch origin` 으로 맞춘 뒤 `git status` 가 깨끗한지 확인.
5. 사용자에게 결과를 한국어로 짧게 알리고, 웹페이지를 고쳤으면 Ctrl+F5 또는 `1_web.bat` 재실행을 안내한다.

### 사용자 응답 방식
- 한국어로 답한다. 질문이 분명하면 바로 답하고 진행한다.
- 코드나 예시를 길게 나열해야 할 때는 먼저 필요한지 물어본다.
- 사용자가 직접 실행해 보고 결과(화면 캡처, 콘솔 로그)를 알려 주는 방식으로 진행한다.

### 남은 일 (2026-11-05 기준)
- 데이터 공유 기능을 사용자 PC에서 실제로 시험 받기 (내보내기 → 다른 PC에서 합치기 → 학습).
- 실시간 인식 화면에 단어와 기여자 표시: 사용자는 원함. 방식(단어별 기여자 목록 / 가장 비슷한 클립의 기여자) 답을 아직 못 받음. 모델 파일에 `contributors` 는 이미 저장됨.
- 내가 지운 남의 클립은 다른 zip으로 다시 들어올 수 있다. 필요하면 지운 해시 목록을 둔다.
- 단어 추가 화면에서 50개 목록 고르기, 단어마다 한국수어사전 링크.
- 사람 단위 검증(기여자를 통째로 떼어 정확도 재기).
- 엄마(mom) 인식 조정: 지금은 잘 안 잡힌다. 사용자에게 실시간 인식의 "최근 3초 최고" 값을 받아 보고,
  녹화 길이 4초로 다시 녹화했는지 확인한 뒤 thresh(0.75 -> 0.65)나 학습 방식을 조정한다.
- 새 폴더 구조(app/, web/, scripts/, data/, models/)로 사용자 PC에서 녹화·학습·인식이 다 되는지 확인 받기.
- 사용자가 원할 때만: 0_setup.bat 끝나면 웹 바로 실행, uv 설치 실패 시 winget/pip로 대신 설치.

### 데이터 공유 형식 (sign_share.py)
- zip: `manifest.json` (format `yolo_finger-signdata`, format_version 1, feat_dim 100, contributor=내보낸 사람, contributors, consent, words, clips) + `clips/<공유단어ID>/<녹화한 사람 ID>__<해시>.npz` (seq, sec, recorded)
- 해시 = float32 seq 바이트의 sha1 앞 16자. 합칠 때 다시 계산하고, 이미 가진 해시(내 클립 포함)는 건너뛴다. 내 ID의 클립은 다시 들여오지 않는다 (지운 내 클립이 돌아오지 않게).
- 내보내기는 내 클립 + 합친 클립 전체를 원래 ID 그대로 넣는다. 압축 안의 이름으로 파일을 풀지 않고 정해진 경로에만 쓴다 (zip slip 방지), 크기·개수 제한 있음.
- 취합 방식 결정(2026-11-05): 참여자가 zip을 슬랙으로 보내고 Langpeu가 웹에서 폴더를 골라 합친다. Hugging Face는 나중에.
- 개발 플랜 문서: https://claude.ai/artifact/3CXA1qF5YrKkWZig9ruREg

### 최근에 한 일
- 데이터 공유 탭 추가: 임시 ID, 내보내기 zip, 폴더 선택 합치기(중복 없음), 합친 데이터 표·기여자 삭제, 학습 결과에 기여자 수. 클립 목록에 기여자 열.
- 공유 단어 50개 목록 `app/vocab.json`.
- 작업 장소를 PC의 git 저장소 `C:\yolo_finger` 로 옮김 (PC에서 커밋, 묶음 파일로 넘겨 클라우드에서 push).
- 종료 버튼: 탭 자동 닫기(window.close) 삭제, 끝 화면 유지, 열린 탭 모두 끝 화면. 탭 잠금 기능은 오작동해서 삭제. 종료 규칙을 고정 규칙으로 문서화.
- 커밋 작성자를 Langpeu(GitHub noreply 주소)로 바꾸고 Claude 공동 작성자 표시를 지움 (기록 다시 쓰고 강제 푸시).
- GitHub 공개(AGPL-3.0), 폴더 정리, Langpeu 로고·이름 표기, 클립·단어 즉시 삭제.

## 꼭 지킬 규칙

### 종료 규칙 - 바꾸지 말 것
README.md "종료 규칙"이 정한 방식을 유지한다. 사용자가 직접 바꾸라고 하기 전에는 고치지 않는다.
- 종료 버튼 -> POST /api/shutdown -> 웹캠 해제 후 os._exit(99) -> 1_web.bat 이 코드 99를 보고 창을 닫음
- 웹페이지에서 window.close() 를 쓰지 않는다. 끝 화면(#byeScreen)을 그대로 둔다.
- 열린 탭 모두 끝 화면으로 바뀐다 (BroadcastChannel 'sign-trainer', type 'quit').
- 탭 잠금 / "다른 탭에서 열려 있어요" 같은 기능을 넣지 않는다.

### 사용자 데이터
- `data/`(녹화 클립·모델·단어·설정·학습 기록)는 사용자 PC 안에서만 처리한다.
  사용자 허락 없이 읽어 가거나 복사하지 않는다. 학습도 사용자 PC에서 한다.
- 테스트는 `SIGN_FAKE_CAM=1` 가짜 카메라로, 사용자 데이터 없이 한다.

### 기타
- 대화와 화면 문구는 한국어. 이름 표기는 **Langpeu** (Lagnepu 아님).
- 경로는 모두 상대 경로(ROOT 기준). 다른 PC로 옮겨도 동작해야 한다.
- `.bat` 파일은 CRLF 줄바꿈 (.gitattributes). py/html/md는 LF.
- 지운 클립·단어는 휴지통 폴더 없이 바로 삭제한다.
- 파일을 고친 뒤에는 `git diff` 로 바뀐 부분을 확인한다.
- 작업을 마칠 때 이 파일의 "남은 일"·"최근에 한 일"을 고쳐서 다음 세션이 이어 갈 수 있게 한다.

## 실행 구조

| 파일 | 역할 |
|---|---|
| `1_web.bat` | `scripts\_env.bat` 로 환경 확인(없으면 설치) 후 `app\web_app.py` 실행. 종료 코드 99면 창을 닫고, 아니면 pause |
| `0_setup.bat` | 환경 설치·복구 (`scripts\_setup.bat called`) |
| `scripts/_env.bat` | ROOT, `ENVDIR=%ROOT%\env_%COMPUTERNAME%`, `OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0` 설정, import 확인 |
| `scripts/_setup.bat` | uv 설치 -> Python 3.12 venv(--clear) -> PyTorch(setup_helper가 고른 버전) -> requirements -> 확인 |
| `app/setup_helper.py` | GPU 확인: compute capability >= 7.5 → cu128, 그보다 오래된 NVIDIA → cu126, 없으면 cpu |
| `app/web_app.py` | 웹 서버(stdlib ThreadingHTTPServer, 127.0.0.1만, Host 헤더 검사), Worker(카메라·녹화·인식), 학습 스레드 |
| `app/sign_common.py` | 경로·설정, 특징 계산, 화자 추적, 모델, 실시간 인식기, 카메라 |
| `app/sign_train.py` | 학습 `train_model(log, progress, stop)`. 기여자별 클립 수를 학습 기록과 모델 파일(`contributors`)에 남김 |
| `app/sign_share.py` | 데이터 공유: 임시 ID, 내보내기 zip, 합치기(중복 검사), 기여자 클립 삭제 |
| `app/vocab.json` | 공유 단어 50개 (+ none). 배고파·목말라·엄마·컵은 빼지 않는다. `cups` 는 `cup` 의 별칭 |
| `web/index.html` | 웹페이지 전체 (반응형, 탭: 녹화 / 실시간 인식 / 학습 / 단어·클립 / 데이터 공유 / 설정 / 학습 기록) |

- 프로젝트는 `C:\yolo_finger` 처럼 C 드라이브 바로 아래에 둔다. 문서·바탕 화면은 윈도우 "제어된 폴더 액세스"가 설치를 막는다.
- `env_*/`, `data/`, `models/*` 는 git에서 제외된다. 옛 폴더 구조(루트의 sign_data 등)는 `_migrate_old_layout()` 이 data/, models/ 로 옮긴다.

### 폴더
- `data/clips/<단어ID>/*` 녹화 클립 (내 클립 `<시각ms>.npz`, 합친 클립 `<임시ID>__<내용해시16>.npz`), `data/contributor.json` 내 임시 ID, `data/sign_model.pt`, `data/words.json`, `data/settings.json`, `data/train_history.json`
- `models/yolo11n-pose.pt`, `models/hand_landmarker.task` (자동 다운로드)

## 웹 API (web_app.py)
- GET: `/`, `/logo.png`, `/favicon.png`, `/frame.jpg?after=N` (한 장씩, 헤더 X-Frame-Id), `/api/state`, `/api/words`, `/api/clips`, `/api/clip`, `/api/train`, `/api/history`, `/api/settings`
- 데이터 공유: GET `/api/share`(현황), GET `/api/export?consent=1`(zip 내려받기), POST `/api/merge?name=` (본문 = zip 바이트), POST `/api/share/id`, POST `/api/share/remove`
- POST: `/api/words/add|update|delete`, `/api/clips/delete` (`{'deleted': n}`), `/api/record/start|stop`, `/api/mode`, `/api/train/start|stop`, `/api/settings`, `/api/shutdown`
- 영상은 MJPEG 스트림 대신 `/frame.jpg` 를 한 장씩 가져온다 (브라우저의 사이트당 연결 6개 제한 때문). 탭이 안 보이면 멈춘다. 클라이언트 fetch에는 타임아웃이 있다.
- 테스트용 환경 변수: `SIGN_FAKE_CAM=1`(웹캠·모델 없이), `SIGN_NO_BROWSER=1`, `SIGN_PORT`.

## 카메라
- `CameraReader` 스레드. MSMF 먼저 -> MSMF 재시도 -> 마지막에 DSHOW. MJPG 요청, 실제 fps 측정, 열기 40초 타임아웃.
- 다음 백엔드를 시도하기 전에 카메라를 반드시 release. 환경 변수는 cv2 import 전에 설정 (안 하면 여는 데 1분 넘게 걸림).
- DSHOW는 YUY2로 열려 10fps가 될 수 있어 쓰지 않는 쪽으로 둔다. 기본 1280x720 30fps (예: Logitech C920).

## 인식 방식
1. **YOLO11n-pose** (GPU): 가장 가까운 사람 = 화자. 다른 사람은 회색 처리, 손목 위치가 안 맞는 손은 버린다 (`SignerTracker.process`).
2. **MediaPipe HandLandmarker** VIDEO 모드 (윈도우에서는 CPU만): 양손 21개 관절.
3. 특징 `FULL_DIM=238`
   - 원시 `FEAT_DIM=100`: 포즈 0,5~10번 + 손 21×2 + 플래그, 어깨 기준 정규화
   - 손 모양: 손목 기준 좌표, 관절 각도, 엄지 끝 거리 (가중치 ×2)
   - 움직임: 손바닥·손목 속도와 빠르기 (가중치 ×4)
4. **GRU SignNet**: hidden 96, 2층, dropout 0.2. 저장 형식 `{state, labels, dim, seq_len, window_sec, hidden}`.

### 학습 (sign_train.py)
- 미니배치 Adam, 클래스 가중치. 실제 클립 단위로 검증용을 떼어 평가한 뒤, 전체 클립으로 최종 모델을 학습한다.
- 증강: 속도 바꾸기(`N_WARP=4`), 정지 반대 예시, 손가락만 바꾼 반대 예시(유사도로 거름), 움직이는 단어(palm_path ≥ 0.8)에서 움직임을 없앤 반대 예시(freeze / line).
- 학습 기록에 "움직임 확인"(motion_check) 값이 남는다. 움직임을 없앤 동작을 그 단어로 잘못 판정한 비율이고, 0%가 좋다.

### 실시간 인식 (Recognizer)
- `add()` 는 매 프레임, `step()` 은 별도 스레드에서 10Hz (`STEP_SEC=0.1`). 그래서 인식 중에도 카메라 fps가 떨어지지 않는다.
- 기본 설정: thresh 0.75, need_frames 3, min_hand 0.3, hold_sec 3.0, cooldown 2.0, window_sec 4.0, fps 12, epochs 300.

## 현재 단어와 진행 상황
- 단어: 배고파(hungry), 목말라(thirsty), 엄마(mom), cups, none(동작 없음).
- (동작 없음)에는 가만히 있기, 손 내려놓기, 손만 들어 올리고 멈추기 같은 평소 동작을 녹화한다.
- 엄마: 손만 올려도 엄마로 판정되던 문제는 움직임 특징으로 고쳤다. 지금은 반대로 엄마가 잘 안 잡히는 편이다.
  녹화 길이 4초로 동작 전체를 다시 녹화하고, "최근 3초 최고" 값을 보며 thresh를 0.65까지 낮춰 보는 중.

## 배포·확인 방법
- 사용자 PC 폴더: `C:\yolo_finger` (git 저장소). 위 "새 세션에서 바로 이어서 하기" 4번 순서로 커밋·push 한다 (Claude 표시 줄 없음).
- 클라우드에서 만든 파일을 PC에 넣을 때 device_commit_files 에 같은 stagedPath를 다시 쓰면 예전 내용이 들어갈 수 있다. 매번 새 경로로 복사하고 md5를 확인한다.
- 웹페이지를 고쳤으면 사용자에게 Ctrl+F5나 `1_web.bat` 재실행을 안내한다.
