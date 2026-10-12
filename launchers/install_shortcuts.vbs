' Point each DogRecognizeTrainer_*.lnk at the windowless .vbs beside this file.
' Run once with: wscript //B //Nologo install_shortcuts.vbs
Option Explicit
Dim sh, fso, folder, names, i, link, vbs, wscriptExe
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
folder = fso.GetParentFolderName(WScript.ScriptFullName)
wscriptExe = sh.ExpandEnvironmentStrings("%SystemRoot%\System32\wscript.exe")
names = Array("DogRecognizeTrainer_sim", "DogRecognizeTrainer_robot", "DogRecognizeTrainer_onboard")
For i = 0 To UBound(names)
  vbs = folder & "\" & names(i) & ".vbs"
  Set link = sh.CreateShortcut(folder & "\" & names(i) & ".lnk")
  link.TargetPath = wscriptExe
  link.Arguments = "//B //Nologo """ & vbs & """"
  link.WorkingDirectory = folder
  link.WindowStyle = 7
  link.Save
Next
