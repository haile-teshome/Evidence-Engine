; Inno Setup script for Evidence Engine (Windows x64).
;
; Produces a single EvidenceEngine-<version>-win-x64-Setup.exe so installing is
; one download and one double-click, instead of unzipping a folder and hunting
; for a .bat file.
;
; Built by packaging/build-windows.ps1, which passes the version and the staged
; payload directory:
;   ISCC.exe /DAppVersion=1.0.0 /DPayloadDir="...\dist-bundle\win-x64\Evidence Engine" installer.iss
;
; It installs per user into %LOCALAPPDATA%\Programs, so it never needs an
; administrator prompt. Nothing is written into the install directory at
; runtime: the venv, the env file and the database live in %USERPROFILE%\.evidence-engine,
; which is why the uninstaller leaves that folder alone and why a reinstall over
; the top keeps every review.

#define AppName "Evidence Engine"
#define AppPublisher "Evidence Engine"
#define AppUrl "https://github.com/haile-teshome/Evidence-Engine"

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef PayloadDir
  #define PayloadDir "..\dist-bundle\win-x64\Evidence Engine"
#endif
#ifndef OutDir
  #define OutDir "..\dist-bundle"
#endif

[Setup]
; Keep this GUID stable forever: it is how Windows recognises an upgrade of this
; same application rather than a second, parallel installation.
AppId={{8E3F4C21-9A7B-4D6E-B1C2-5F0A7D3E9B14}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppUrl}
AppSupportURL={#AppUrl}/issues
AppUpdatesURL={#AppUrl}/releases
VersionInfoVersion={#AppVersion}

DefaultDirName={localappdata}\Programs\Evidence Engine
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

OutputDir={#OutDir}
OutputBaseFilename=EvidenceEngine-{#AppVersion}-win-x64-Setup
SetupIconFile={#PayloadDir}\assets\icon.ico
UninstallDisplayIcon={app}\assets\icon.ico
UninstallDisplayName={#AppName}
WizardStyle=modern
; The payload is large and mostly already-compressed wheels and runtimes.
Compression=lzma2/max
SolidCompression=yes
LicenseFile={#PayloadDir}\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"

[Files]
Source: "{#PayloadDir}\*"; DestDir: "{app}"; \
  Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
; Both shortcuts run the hidden launcher so there is no console window. The
; launcher opens the app window itself and shuts down when that window closes.
Name: "{group}\{#AppName}"; Filename: "{sys}\wscript.exe"; \
  Parameters: """{app}\launch-hidden.vbs"""; WorkingDir: "{app}"; \
  IconFilename: "{app}\assets\icon.ico"
Name: "{autodesktop}\{#AppName}"; Filename: "{sys}\wscript.exe"; \
  Parameters: """{app}\launch-hidden.vbs"""; WorkingDir: "{app}"; \
  IconFilename: "{app}\assets\icon.ico"; Tasks: desktopicon
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

[Run]
Filename: "{sys}\wscript.exe"; Parameters: """{app}\launch-hidden.vbs"""; \
  WorkingDir: "{app}"; Description: "Open {#AppName} now"; \
  Flags: nowait postinstall skipifsilent

[Messages]
; Set expectations before the first launch rather than after it.
FinishedLabel=Evidence Engine is installed.%n%nThe first launch takes a few minutes: it builds its own Python environment and downloads the local AI model (about 4 GB). After that it starts in seconds.%n%nYour reviews are saved in your user folder, not inside the app, so you can install a newer version over this one and keep everything.
