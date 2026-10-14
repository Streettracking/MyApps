' Windowless YOLO teacher. Style 0, pythonw when the venv has it.
' The trainer also starts this service when port 8091 is down, and stops that copy on exit.
Option Explicit
Dim sh, fso, user, py, weights, serve, cmd
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
user = sh.ExpandEnvironmentStrings("%USERPROFILE%")
py = user & "\Desktop\FlyWire-Go2\yolo\venv\Scripts\pythonw.exe"
If Not fso.FileExists(py) Then py = user & "\Desktop\FlyWire-Go2\yolo\venv\Scripts\python.exe"
weights = user & "\Desktop\FlyWire-Go2\yolo\runs\go2_mix_v2\weights\best.pt"
serve = fso.GetParentFolderName(WScript.ScriptFullName) & "\serve.py"
cmd = Chr(34) & py & Chr(34) & " " & Chr(34) & serve & Chr(34) & " --weights " & Chr(34) & weights & Chr(34) & " --host 127.0.0.1 --port 8091"
sh.Run cmd, 0, False
