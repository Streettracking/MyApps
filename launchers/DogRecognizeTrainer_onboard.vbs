' Windowless onboard launcher. wscript itself has no console.
' Style 0 hides cmd.exe. The trainer exe is built --noconsole and opens its own window.
Option Explicit
Dim sh, fso, folder, exe, cmd
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
exe = folder & "\DogRecognizeTrainer.exe"
If Not fso.FileExists(exe) Then exe = folder & "\recognize_trainer.exe"
cmd = "cmd.exe /c " & Chr(34) & Chr(34) & exe & Chr(34) & " --onboard 192.168.35.213" & Chr(34)
sh.Run cmd, 0, False
