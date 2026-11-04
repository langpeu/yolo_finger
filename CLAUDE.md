# 작업 규칙

## 종료 규칙 - 바꾸지 말 것
README.md "종료 규칙"이 정한 방식을 유지한다. 사용자가 직접 바꾸라고 하기 전에는 고치지 않는다.
- 종료 버튼 -> POST /api/shutdown -> 웹캠 해제 후 os._exit(99) -> 1_web.bat 이 코드 99를 보고 창을 닫음
- 웹페이지에서 window.close() 를 쓰지 않는다. 끝 화면(#byeScreen)을 그대로 둔다.
- 열린 탭 모두 끝 화면으로 바뀐다 (BroadcastChannel 'sign-trainer', type 'quit').
- 탭 잠금 / "다른 탭에서 열려 있어요" 같은 기능을 넣지 않는다.
