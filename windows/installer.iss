; Inno Setup script for the Pitch Speed installer (free; built by GitHub Actions).
; Installs for the current user only (no administrator rights needed).
#define AppVersion GetEnv("PITCHSPEED_VERSION")
#if AppVersion == ""
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{6B0E3F4A-5C1D-4E8A-9F3B-2A7C8D9E0F11}
AppName=Pitch Speed
AppVersion={#AppVersion}
AppPublisher=Pitch Speed
AppPublisherURL=https://github.com/Skewtoob/Pitch-Speed-Estimator
DefaultDirName={localappdata}\Programs\Pitch Speed
DefaultGroupName=Pitch Speed
PrivilegesRequired=lowest
OutputDir=..\installer
OutputBaseFilename=PitchSpeed-Setup
SetupIconFile=icon.ico
UninstallDisplayIcon={app}\PitchSpeed.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes

[Tasks]
Name: "desktopicon"; Description: "Put a Pitch Speed icon on the desktop"; GroupDescription: "Shortcuts:"

[Files]
Source: "..\dist\PitchSpeed\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\Pitch Speed"; Filename: "{app}\PitchSpeed.exe"
Name: "{group}\Pitch Speed instructions"; Filename: "{app}\README.md"
Name: "{group}\Uninstall Pitch Speed"; Filename: "{uninstallexe}"
Name: "{userdesktop}\Pitch Speed"; Filename: "{app}\PitchSpeed.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\PitchSpeed.exe"; Description: "Start Pitch Speed now"; Flags: nowait postinstall skipifsilent
