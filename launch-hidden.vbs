' Starts Evidence Engine with no console window. This is what the Start Menu and
' desktop shortcuts created by the Windows installer point at.
'
' launch.bat stays the visible route: it prints progress and waits at the end,
' which is what you want when something has gone wrong. That trailing wait is
' also why the shortcuts cannot just run launch.bat hidden, since the invisible
' window would sit there waiting for a key press nobody can press.
Option Explicit

Dim fso, sh, base, node, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")

base = fso.GetParentFolderName(WScript.ScriptFullName)

node = base & "\runtime\node\node.exe"            ' the bundled runtime
If Not fso.FileExists(node) Then node = "node"    ' else a system install

cmd = """" & node & """ """ & base & "\launch.mjs"""
sh.CurrentDirectory = base
sh.Run cmd, 0, False                              ' 0 = hidden window, False = do not wait
