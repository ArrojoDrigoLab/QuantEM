; QuantEM NSIS installer hooks (bundle.windows.nsis.installerHooks).
;
; Two jobs (owner ruling 2026-08-09, "model downloads at install time"):
;
;   1. A "Model downloads" page -- eight checkboxes, one per model pack, the
;      four OmniEM packs checked by default -- shown right after the install
;      directory page. The page only records a selection; NOTHING is
;      downloaded by the installer. On first launch the app reads the pending
;      file written below and queues the installs through its own verified
;      download machinery (digest checks, progress, cancel, AV-retry).
;
;   2. NSIS_HOOK_POSTINSTALL writes that selection to
;          <INSTDIR>\data\pending-model-installs.json
;      as {"packs": ["omniem:mito", ...]} -- the contract with the server's
;      first-launch consumer. <INSTDIR>\data is the app's data directory:
;      the shell defaults QUANTEM_DATA_DIR to <exe dir>\data.
;
; This file is !include'd at the TOP of the installer script, before the
; template declares its variables ($PassiveMode, $UpdateMode) -- so only Var
; declarations and macro DEFINITIONS live at include scope. The page functions
; are wrapped in QUANTEM_MODEL_PAGE_FUNCTIONS, which the vendored template
; (nsis/installer.nsi) expands between the directory and start-menu pages.
;
; Pack sizes shown to the user are the real published Hugging Face sizes,
; transcribed from MEASURED_SIZES in src/quantem/registry/manifest.py
; (decimal MB, matching the Models screen):
;
;   omniem_vitl_encoder   1,217,509,768 B -> 1218 MB (shared OmniEM trunk)
;   quantem_vitb_encoder    227,685,512 B ->  228 MB (shared QuantEM trunk)
;   mito_omniem_head         25,730,696 B ->   26 MB
;   er_omniem_head          135,200,976 B ->  135 MB
;   nucleus_omniem_head      25,730,704 B ->   26 MB
;   ld_omniem_head           25,730,688 B ->   26 MB
;   mito_quantem_head       136,541,856 B ->  137 MB
;   er_quantem_head         465,028,184 B ->  465 MB
;   nucleus_quantem_head    136,541,864 B ->  137 MB
;   ld_quantem_head         136,541,848 B ->  137 MB
;
; A family's trunk downloads once, with the first pack of that family -- the
; running total below accounts for that (head sizes + each family's trunk
; counted once), so the number the user sees is the number that downloads.

!include "nsDialogs.nsh"
!include "${__FILEDIR__}\payload-manifest.nsh"

; ---- Storage-model copy (uat13, round 14) ----------------------------------
; 1. Directory chooser top text. MUI_PAGE_DIRECTORY (inserted later by the
;    vendored template) consumes MUI_DIRECTORYPAGE_TEXT_TOP if it is already
;    defined, so defining it here needs no template change. $(^DirText) is the
;    stock localized explanation; one sentence about the storage model is
;    appended. The page's text control (id 1006 in modern.exe) is 60 DLU tall
;    (~7 text lines), so the combined ~6 lines render without clipping.
!define MUI_DIRECTORYPAGE_TEXT_TOP "$(^DirText)$\r$\n$\r$\nQuantEM stores all of its data - including multi-GB model downloads - under the folder you choose, so pick a location with enough free space and avoid OneDrive, Dropbox or other cloud-synced folders."

; 2. Uninstaller "delete data" checkbox copy: must name what it deletes. The
;    stock $(deleteAppData) LangString cannot be overridden from here (the
;    bundler's language files define it after this file is included), so the
;    vendored template's un.ConfirmShow uses this define instead -- see the
;    fenced QuantEM insertion there. __NSD_CheckBox_STYLE already carries
;    BS_MULTILINE; the template insertion also makes the control tall enough
;    for the wrapped text.
!define QUANTEM_DELETE_DATA_TEXT "Delete the data\ folder in the install directory: imported images, proofreading work, analyses, exports and downloaded models (can be several GB)."

Var QMdlDialog
Var QMdlTotalLabel
; Checkbox control handles.
Var QMdlCbOmniMito
Var QMdlCbOmniEr
Var QMdlCbOmniNuc
Var QMdlCbOmniLd
Var QMdlCbQemMito
Var QMdlCbQemEr
Var QMdlCbQemNuc
Var QMdlCbQemLd
; Checkbox states (persist across Back/Next; read by the POSTINSTALL hook).
Var QMdlStOmniMito
Var QMdlStOmniEr
Var QMdlStOmniNuc
Var QMdlStOmniLd
Var QMdlStQemMito
Var QMdlStQemEr
Var QMdlStQemNuc
Var QMdlStQemLd
; 1 once the user has actually been shown the page and moved past it. A
; passive or update install never shows the page and must not invent or
; clobber a selection.
Var QMdlConfigured

; ---- CPU/CUDA runtime selection and layered installation -------------------
; The signed installer carries the small application layers for both runtime
; variants. CI publishes the large, immutable Python/PyTorch runtimes as
; hash-pinned release assets. Routine upgrades retain a content-identical
; runtime and replace only the embedded application layer.
Var QRuntimeDialog
Var QRuntimeCpuRadio
Var QRuntimeCudaRadio
Var QRuntimeStatusLabel
Var QRuntimeVariant
Var QRuntimeInitialized
Var QRuntimeCudaDetected
Var QRuntimeCudaDriverApi
Var QRuntimeUrl1
Var QRuntimeUrl2
Var QRuntimeUrl3
Var QRuntimeUrl4
Var QRuntimeHash1
Var QRuntimeHash2
Var QRuntimeHash3
Var QRuntimeHash4
Var QRuntimePartCount
Var QRuntimeArchiveHash
Var QRuntimeArchive
Var QRuntimeStage
Var QRuntimeBackup
Var QRuntimeRequiredId
Var QRuntimeCurrent
Var QRuntimeChanged
Var QApplicationArchiveHash
Var QApplicationArchive
Var QApplicationStage
Var QApplicationBackup
Var QDownloadUrl
Var QDownloadHash
Var QDownloadPath
Var QVerifyPath
Var QVerifyHash
Var QVerifyOk

!macro QUANTEM_RUNTIME_PAGE_FUNCTIONS

Function QuantemDetectCuda
  StrCpy $QRuntimeCudaDetected 0
  StrCpy $QRuntimeCudaDriverApi 0
  System::Call 'kernel32::LoadLibraryW(w "nvcuda.dll") p .r0'
  ${If} $0 = 0
    Return
  ${EndIf}
  System::Call 'kernel32::GetProcAddress(p r0, m "cuInit") p .r1'
  ${If} $1 = 0
    System::Call 'kernel32::FreeLibrary(p r0)'
    Return
  ${EndIf}
  System::Call '::$1(i 0) i .r2'
  ${If} $2 <> 0
    System::Call 'kernel32::FreeLibrary(p r0)'
    Return
  ${EndIf}
  System::Call 'kernel32::GetProcAddress(p r0, m "cuDriverGetVersion") p .r1'
  ${If} $1 <> 0
    System::Call '::$1(*i .r3) i .r2'
    ${If} $2 = 0
      StrCpy $QRuntimeCudaDriverApi $3
      ${If} $3 >= ${QPAYLOAD_CUDA_MIN_DRIVER_API}
        StrCpy $QRuntimeCudaDetected 1
      ${EndIf}
    ${EndIf}
  ${EndIf}
  System::Call 'kernel32::FreeLibrary(p r0)'
FunctionEnd

Function QuantemInitializeRuntimeChoice
  ${If} $QRuntimeInitialized = 1
    Return
  ${EndIf}
  StrCpy $QRuntimeInitialized 1
  StrCpy $QRuntimeVariant "cpu"

  ; Updates always preserve the installed flavor. A legacy installation has
  ; no marker and safely enters the new scheme as CPU.
  ${If} $UpdateMode = 1
    StrCpy $0 ""
    ClearErrors
    FileOpen $1 "$INSTDIR\.quantem-runtime-variant" r
    ${IfNot} ${Errors}
      FileRead $1 $0
      FileClose $1
    ${EndIf}
    ${If} $0 == ""
      ReadRegStr $0 SHCTX "${MANUPRODUCTKEY}" "RuntimeVariant"
    ${EndIf}
    ${If} $0 == "cuda"
      StrCpy $QRuntimeVariant "cuda"
    ${EndIf}
    Return
  ${EndIf}

  ; An explicit flag is useful for managed/silent installs and CI smoke tests.
  ClearErrors
  ${GetOptions} $CMDLINE "/QUANTEM_VARIANT=" $0
  ${IfNot} ${Errors}
    ${If} $0 == "cpu"
    ${OrIf} $0 == "cuda"
      StrCpy $QRuntimeVariant $0
      Return
    ${EndIf}
  ${EndIf}

  Call QuantemDetectCuda
  ${If} $QRuntimeCudaDetected = 1
    StrCpy $QRuntimeVariant "cuda"
  ${EndIf}
FunctionEnd

Function PageQuantemRuntime
  ${If} $PassiveMode = 1
  ${OrIf} $UpdateMode = 1
    Abort
  ${EndIf}
  Call QuantemInitializeRuntimeChoice

  !insertmacro MUI_HEADER_TEXT "Hardware acceleration" "Choose the runtime installed inside QuantEM."
  nsDialogs::Create 1018
  Pop $QRuntimeDialog
  ${If} $QRuntimeDialog == error
    Abort
  ${EndIf}

  ${NSD_CreateLabel} 0 0 100% 30u "QuantEM checked your NVIDIA driver and selected the recommended runtime. You can change it here; this choice is retained by future automatic updates."
  Pop $0
  ${NSD_CreateRadioButton} 0 36u 100% 16u "NVIDIA CUDA acceleration (${QPAYLOAD_CUDA_SIZE_MB} MB download)"
  Pop $QRuntimeCudaRadio
  ${NSD_CreateRadioButton} 0 58u 100% 16u "CPU (${QPAYLOAD_CPU_SIZE_MB} MB download; works on every supported Windows PC)"
  Pop $QRuntimeCpuRadio
  ${NSD_CreateLabel} 0 82u 100% 36u ""
  Pop $QRuntimeStatusLabel

  Call QuantemDetectCuda
  ${If} $QRuntimeCudaDetected = 1
    IntOp $0 $QRuntimeCudaDriverApi / 1000
    IntOp $1 $QRuntimeCudaDriverApi % 1000
    IntOp $1 $1 / 10
    ${NSD_SetText} $QRuntimeStatusLabel "Compatible NVIDIA CUDA driver detected (CUDA API $0.$1). CUDA is recommended."
  ${ElseIf} $QRuntimeCudaDriverApi > 0
    ${NSD_SetText} $QRuntimeStatusLabel "An NVIDIA driver was found, but it is too old for this CUDA runtime. CPU is recommended unless you update the driver."
  ${Else}
    ${NSD_SetText} $QRuntimeStatusLabel "No compatible NVIDIA CUDA driver was detected. CPU is recommended."
  ${EndIf}

  ${If} $QRuntimeVariant == "cuda"
    ${NSD_Check} $QRuntimeCudaRadio
  ${Else}
    ${NSD_Check} $QRuntimeCpuRadio
  ${EndIf}
  nsDialogs::Show
FunctionEnd

Function PageLeaveQuantemRuntime
  ${NSD_GetState} $QRuntimeCudaRadio $0
  ${If} $0 = ${BST_CHECKED}
    StrCpy $QRuntimeVariant "cuda"
  ${Else}
    StrCpy $QRuntimeVariant "cpu"
  ${EndIf}
FunctionEnd

Function QuantemSelectPayload
  StrCpy $QRuntimeUrl1 ""
  StrCpy $QRuntimeUrl2 ""
  StrCpy $QRuntimeUrl3 ""
  StrCpy $QRuntimeUrl4 ""
  StrCpy $QRuntimeHash1 ""
  StrCpy $QRuntimeHash2 ""
  StrCpy $QRuntimeHash3 ""
  StrCpy $QRuntimeHash4 ""
  ${If} $QRuntimeVariant == "cuda"
    StrCpy $QRuntimeRequiredId "${QPAYLOAD_CUDA_RUNTIME_ID}"
    StrCpy $QApplicationArchiveHash "${QPAYLOAD_CUDA_APPLICATION_SHA256}"
    StrCpy $QRuntimePartCount ${QPAYLOAD_CUDA_PART_COUNT}
    StrCpy $QRuntimeArchiveHash "${QPAYLOAD_CUDA_SHA256}"
    StrCpy $QRuntimeUrl1 "${QPAYLOAD_CUDA_PART1_URL}"
    StrCpy $QRuntimeHash1 "${QPAYLOAD_CUDA_PART1_SHA256}"
    !if ${QPAYLOAD_CUDA_PART_COUNT} >= 2
      StrCpy $QRuntimeUrl2 "${QPAYLOAD_CUDA_PART2_URL}"
      StrCpy $QRuntimeHash2 "${QPAYLOAD_CUDA_PART2_SHA256}"
    !endif
    !if ${QPAYLOAD_CUDA_PART_COUNT} >= 3
      StrCpy $QRuntimeUrl3 "${QPAYLOAD_CUDA_PART3_URL}"
      StrCpy $QRuntimeHash3 "${QPAYLOAD_CUDA_PART3_SHA256}"
    !endif
    !if ${QPAYLOAD_CUDA_PART_COUNT} >= 4
      StrCpy $QRuntimeUrl4 "${QPAYLOAD_CUDA_PART4_URL}"
      StrCpy $QRuntimeHash4 "${QPAYLOAD_CUDA_PART4_SHA256}"
    !endif
  ${Else}
    StrCpy $QRuntimeRequiredId "${QPAYLOAD_CPU_RUNTIME_ID}"
    StrCpy $QApplicationArchiveHash "${QPAYLOAD_CPU_APPLICATION_SHA256}"
    StrCpy $QRuntimePartCount ${QPAYLOAD_CPU_PART_COUNT}
    StrCpy $QRuntimeArchiveHash "${QPAYLOAD_CPU_SHA256}"
    StrCpy $QRuntimeUrl1 "${QPAYLOAD_CPU_PART1_URL}"
    StrCpy $QRuntimeHash1 "${QPAYLOAD_CPU_PART1_SHA256}"
    !if ${QPAYLOAD_CPU_PART_COUNT} >= 2
      StrCpy $QRuntimeUrl2 "${QPAYLOAD_CPU_PART2_URL}"
      StrCpy $QRuntimeHash2 "${QPAYLOAD_CPU_PART2_SHA256}"
    !endif
    !if ${QPAYLOAD_CPU_PART_COUNT} >= 3
      StrCpy $QRuntimeUrl3 "${QPAYLOAD_CPU_PART3_URL}"
      StrCpy $QRuntimeHash3 "${QPAYLOAD_CPU_PART3_SHA256}"
    !endif
    !if ${QPAYLOAD_CPU_PART_COUNT} >= 4
      StrCpy $QRuntimeUrl4 "${QPAYLOAD_CPU_PART4_URL}"
      StrCpy $QRuntimeHash4 "${QPAYLOAD_CPU_PART4_SHA256}"
    !endif
  ${EndIf}
FunctionEnd

Function QuantemVerifyFile
  StrCpy $QVerifyOk 0
  ; Windows PowerShell's Get-FileHash cmdlet is not present in every minimal
  ; Windows image. The underlying .NET SHA-256 API is part of every supported
  ; Windows PowerShell runtime and avoids that optional-module dependency.
  nsExec::ExecToStack '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "if ([String]::Equals([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash([IO.File]::OpenRead($\'$QVerifyPath$\'))).Replace($\'-$\',$\'$\'),$\'$QVerifyHash$\',[StringComparison]::OrdinalIgnoreCase)) { exit 0 } else { exit 1 }"'
  Pop $0
  Pop $1
  ${If} $0 = 0
    StrCpy $QVerifyOk 1
  ${Else}
    DetailPrint "SHA-256 verification failed: $1"
    FileOpen $R8 "$INSTDIR\.quantem-install\failure.log" a
    FileWrite $R8 "SHA-256 verification command failed ($0): $1$\r$\n"
    FileClose $R8
  ${EndIf}
FunctionEnd

Function QuantemDownloadPart
  quantem_download_retry:
  Delete "$QDownloadPath"
  DetailPrint "Downloading $QRuntimeVariant runtime: $QDownloadUrl"
  nsExec::ExecToStack '"$SYSDIR\curl.exe" --fail --location --retry 3 --connect-timeout 30 --silent --show-error --output "$QDownloadPath" "$QDownloadUrl"'
  Pop $0
  Pop $1
  ; curl.exe is part of current Windows, but 32-bit process redirection can
  ; hide it on some installations. Windows PowerShell is the compatibility
  ; fallback and writes to the same install-drive staging directory.
  ${If} $0 <> 0
    DetailPrint "Native curl was unavailable or failed; retrying with Windows PowerShell."
    nsExec::ExecToStack '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command "Invoke-WebRequest -UseBasicParsing -Uri $\'$QDownloadUrl$\' -OutFile $\'$QDownloadPath$\'"'
    Pop $0
    Pop $1
  ${EndIf}
  ${If} $0 <> 0
    DetailPrint "Runtime download failed: $1"
    FileOpen $R8 "$INSTDIR\.quantem-install\failure.log" a
    FileWrite $R8 "Runtime download failed ($0): $1$\r$\n"
    FileClose $R8
    ${If} ${Silent}
      Abort "The QuantEM runtime download failed."
    ${EndIf}
    MessageBox MB_ICONEXCLAMATION|MB_RETRYCANCEL "The QuantEM runtime download failed.$\r$\n$\r$\nCheck your internet connection, then click Retry." IDRETRY quantem_download_retry
    Abort "The QuantEM runtime download was cancelled."
  ${EndIf}
  StrCpy $QVerifyPath "$QDownloadPath"
  StrCpy $QVerifyHash "$QDownloadHash"
  Call QuantemVerifyFile
  ${If} $QVerifyOk <> 1
    Delete "$QDownloadPath"
    ${If} ${Silent}
      Abort "The downloaded QuantEM runtime failed its security check."
    ${EndIf}
    MessageBox MB_ICONSTOP|MB_RETRYCANCEL "The downloaded runtime failed its SHA-256 security check.$\r$\n$\r$\nClick Retry to download it again." IDRETRY quantem_download_retry
    Abort "The QuantEM runtime download was cancelled."
  ${EndIf}
FunctionEnd

Function QuantemStageApplication
  StrCpy $QApplicationArchive "$INSTDIR\.quantem-install\application.zip"
  StrCpy $QApplicationStage "$INSTDIR\.quantem-install\application"
  StrCpy $QApplicationBackup "$INSTDIR\.quantem-install\old-application"
  CreateDirectory "$QApplicationStage"
  SetOutPath "$INSTDIR\.quantem-install"
  ${If} $QRuntimeVariant == "cuda"
    File /oname=application.zip "${QPAYLOAD_CUDA_APPLICATION_PATH}"
  ${Else}
    File /oname=application.zip "${QPAYLOAD_CPU_APPLICATION_PATH}"
  ${EndIf}
  File /oname=verify-existing-runtime.ps1 "${QPAYLOAD_RUNTIME_VERIFIER_PATH}"
  SetOutPath "$INSTDIR"

  StrCpy $QVerifyPath "$QApplicationArchive"
  StrCpy $QVerifyHash "$QApplicationArchiveHash"
  Call QuantemVerifyFile
  ${If} $QVerifyOk <> 1
    Abort "The embedded QuantEM application layer failed its security check."
  ${EndIf}
  nsExec::ExecToStack '"$SYSDIR\tar.exe" -xf "$QApplicationArchive" -C "$QApplicationStage"'
  Pop $0
  Pop $1
  ${If} $0 <> 0
    Abort "The QuantEM application layer could not be extracted."
  ${EndIf}
  IfFileExists "$QApplicationStage\quantem-server\quantem-server.exe" 0 application_incomplete
  IfFileExists "$QApplicationStage\quantem-server\build-info.json" 0 application_incomplete
  IfFileExists "$QApplicationStage\quantem-layer\runtime-files.json" 0 application_incomplete
  IfFileExists "$QApplicationStage\quantem-layer\runtime-info.json" 0 application_incomplete
  Return
  application_incomplete:
    Abort "The verified QuantEM application layer is incomplete."
FunctionEnd

Function QuantemCheckInstalledRuntime
  StrCpy $QRuntimeCurrent 0
  StrCpy $0 ""
  ClearErrors
  FileOpen $1 "$INSTDIR\.quantem-runtime-id" r
  ${IfNot} ${Errors}
    FileRead $1 $0
    FileClose $1
  ${EndIf}
  ${If} $0 == $QRuntimeRequiredId
    IfFileExists "$INSTDIR\quantem-server\runtime-info.json" 0 runtime_check_done
    StrCpy $QRuntimeCurrent 1
    DetailPrint "Retaining compatible $QRuntimeVariant runtime $QRuntimeRequiredId."
    Return
  ${EndIf}

  ; First layered upgrade: an older monolithic installation has no runtime-ID
  ; marker. Hash only the required runtime files against the embedded manifest;
  ; if every byte matches, adopt it without downloading PyTorch again.
  IfFileExists "$INSTDIR\quantem-server\_internal\*.*" 0 runtime_check_done
  nsExec::ExecToStack '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "$INSTDIR\.quantem-install\verify-existing-runtime.ps1" -Root "$INSTDIR\quantem-server" -Manifest "$QApplicationStage\quantem-layer\runtime-files.json"'
  Pop $0
  Pop $1
  ${If} $0 = 0
    ClearErrors
    CopyFiles /SILENT "$QApplicationStage\quantem-layer\runtime-info.json" "$INSTDIR\quantem-server"
    ${If} ${Errors}
      DetailPrint "The existing runtime matched, but its compatibility marker could not be installed."
    ${Else}
      StrCpy $QRuntimeCurrent 1
      DetailPrint "Adopted the existing verified $QRuntimeVariant runtime; no runtime download is needed."
    ${EndIf}
  ${Else}
    DetailPrint "Installed runtime is not byte-compatible with $QRuntimeRequiredId; a replacement is required."
  ${EndIf}
  runtime_check_done:
FunctionEnd

Function QuantemBackupApplication
  CreateDirectory "$QApplicationBackup\_internal"
  Rename "$INSTDIR\quantem-server\quantem-server.exe" "$QApplicationBackup\quantem-server.exe"
  Rename "$INSTDIR\quantem-server\build-info.json" "$QApplicationBackup\build-info.json"
  Rename "$INSTDIR\quantem-server\_internal\quantem" "$QApplicationBackup\_internal\quantem"
  Rename "$INSTDIR\quantem-server\_internal\quantem_frontend" "$QApplicationBackup\_internal\quantem_frontend"
  FindFirst $0 $1 "$INSTDIR\quantem-server\_internal\quantem_app-*.dist-info"
  application_backup_dist_info:
    StrCmp $1 "" application_backup_done
    Rename "$INSTDIR\quantem-server\_internal\$1" "$QApplicationBackup\_internal\$1"
    FindNext $0 $1
    Goto application_backup_dist_info
  application_backup_done:
  FindClose $0
FunctionEnd

Function QuantemRemoveApplication
  Delete "$INSTDIR\quantem-server\quantem-server.exe"
  Delete "$INSTDIR\quantem-server\build-info.json"
  RMDir /r "$INSTDIR\quantem-server\_internal\quantem"
  RMDir /r "$INSTDIR\quantem-server\_internal\quantem_frontend"
  FindFirst $0 $1 "$INSTDIR\quantem-server\_internal\quantem_app-*.dist-info"
  application_remove_dist_info:
    StrCmp $1 "" application_remove_done
    RMDir /r "$INSTDIR\quantem-server\_internal\$1"
    FindNext $0 $1
    Goto application_remove_dist_info
  application_remove_done:
  FindClose $0
FunctionEnd

Function QuantemRestoreApplication
  Call QuantemRemoveApplication
  Rename "$QApplicationBackup\quantem-server.exe" "$INSTDIR\quantem-server\quantem-server.exe"
  Rename "$QApplicationBackup\build-info.json" "$INSTDIR\quantem-server\build-info.json"
  Rename "$QApplicationBackup\_internal\quantem" "$INSTDIR\quantem-server\_internal\quantem"
  Rename "$QApplicationBackup\_internal\quantem_frontend" "$INSTDIR\quantem-server\_internal\quantem_frontend"
  FindFirst $0 $1 "$QApplicationBackup\_internal\quantem_app-*.dist-info"
  application_restore_dist_info:
    StrCmp $1 "" application_restore_done
    Rename "$QApplicationBackup\_internal\$1" "$INSTDIR\quantem-server\_internal\$1"
    FindNext $0 $1
    Goto application_restore_dist_info
  application_restore_done:
  FindClose $0
FunctionEnd

Function QuantemPromoteApplication
  StrCpy $R7 0
  ClearErrors
  Rename "$QApplicationStage\quantem-server\quantem-server.exe" "$INSTDIR\quantem-server\quantem-server.exe"
  ${If} ${Errors}
    StrCpy $R7 1
  ${EndIf}
  ClearErrors
  Rename "$QApplicationStage\quantem-server\build-info.json" "$INSTDIR\quantem-server\build-info.json"
  ${If} ${Errors}
    StrCpy $R7 1
  ${EndIf}
  IfFileExists "$QApplicationStage\quantem-server\_internal\quantem\*.*" 0 application_promote_frontend
    ClearErrors
    Rename "$QApplicationStage\quantem-server\_internal\quantem" "$INSTDIR\quantem-server\_internal\quantem"
    ${If} ${Errors}
      StrCpy $R7 1
    ${EndIf}
  application_promote_frontend:
  IfFileExists "$QApplicationStage\quantem-server\_internal\quantem_frontend\*.*" 0 application_promote_dist_info_start
    ClearErrors
    Rename "$QApplicationStage\quantem-server\_internal\quantem_frontend" "$INSTDIR\quantem-server\_internal\quantem_frontend"
    ${If} ${Errors}
      StrCpy $R7 1
    ${EndIf}
  application_promote_dist_info_start:
  FindFirst $0 $1 "$QApplicationStage\quantem-server\_internal\quantem_app-*.dist-info"
  application_promote_dist_info:
    StrCmp $1 "" application_promote_done
    ClearErrors
    Rename "$QApplicationStage\quantem-server\_internal\$1" "$INSTDIR\quantem-server\_internal\$1"
    ${If} ${Errors}
      StrCpy $R7 1
    ${EndIf}
    FindNext $0 $1
    Goto application_promote_dist_info
  application_promote_done:
  FindClose $0
FunctionEnd

Function QuantemInstallRuntime
  Call QuantemInitializeRuntimeChoice
  Call QuantemSelectPayload
  StrCpy $QRuntimeArchive "$INSTDIR\.quantem-install\runtime.zip"
  StrCpy $QRuntimeStage "$INSTDIR\.quantem-install\new-runtime"
  StrCpy $QRuntimeBackup "$INSTDIR\.quantem-install\old-runtime"
  StrCpy $QRuntimeChanged 0
  RMDir /r "$INSTDIR\.quantem-install"
  CreateDirectory "$QRuntimeStage"
  Call QuantemStageApplication
  Call QuantemCheckInstalledRuntime
  ${If} $QRuntimeCurrent = 1
    Goto runtime_ready
  ${EndIf}

  StrCpy $QDownloadUrl $QRuntimeUrl1
  StrCpy $QDownloadHash $QRuntimeHash1
  StrCpy $QDownloadPath "$INSTDIR\.quantem-install\part01"
  Call QuantemDownloadPart
  ${If} $QRuntimePartCount >= 2
    StrCpy $QDownloadUrl $QRuntimeUrl2
    StrCpy $QDownloadHash $QRuntimeHash2
    StrCpy $QDownloadPath "$INSTDIR\.quantem-install\part02"
    Call QuantemDownloadPart
  ${EndIf}
  ${If} $QRuntimePartCount >= 3
    StrCpy $QDownloadUrl $QRuntimeUrl3
    StrCpy $QDownloadHash $QRuntimeHash3
    StrCpy $QDownloadPath "$INSTDIR\.quantem-install\part03"
    Call QuantemDownloadPart
  ${EndIf}
  ${If} $QRuntimePartCount >= 4
    StrCpy $QDownloadUrl $QRuntimeUrl4
    StrCpy $QDownloadHash $QRuntimeHash4
    StrCpy $QDownloadPath "$INSTDIR\.quantem-install\part04"
    Call QuantemDownloadPart
  ${EndIf}

  ${If} $QRuntimePartCount = 1
    Rename "$INSTDIR\.quantem-install\part01" "$QRuntimeArchive"
  ${ElseIf} $QRuntimePartCount = 2
    nsExec::ExecToStack '"$SYSDIR\cmd.exe" /D /C copy /B "$INSTDIR\.quantem-install\part01"+"$INSTDIR\.quantem-install\part02" "$QRuntimeArchive" >NUL'
    Pop $0
    Pop $1
  ${ElseIf} $QRuntimePartCount = 3
    nsExec::ExecToStack '"$SYSDIR\cmd.exe" /D /C copy /B "$INSTDIR\.quantem-install\part01"+"$INSTDIR\.quantem-install\part02"+"$INSTDIR\.quantem-install\part03" "$QRuntimeArchive" >NUL'
    Pop $0
    Pop $1
  ${Else}
    nsExec::ExecToStack '"$SYSDIR\cmd.exe" /D /C copy /B "$INSTDIR\.quantem-install\part01"+"$INSTDIR\.quantem-install\part02"+"$INSTDIR\.quantem-install\part03"+"$INSTDIR\.quantem-install\part04" "$QRuntimeArchive" >NUL'
    Pop $0
    Pop $1
  ${EndIf}

  StrCpy $QVerifyPath "$QRuntimeArchive"
  StrCpy $QVerifyHash "$QRuntimeArchiveHash"
  Call QuantemVerifyFile
  ${If} $QVerifyOk <> 1
    Abort "The assembled QuantEM runtime failed its security check."
  ${EndIf}
  DetailPrint "Extracting the verified $QRuntimeVariant runtime..."
  nsExec::ExecToStack '"$SYSDIR\tar.exe" -xf "$QRuntimeArchive" -C "$QRuntimeStage"'
  Pop $0
  Pop $1
  ${If} $0 <> 0
    Abort "The QuantEM runtime could not be extracted."
  ${EndIf}
  IfFileExists "$QRuntimeStage\quantem-server\runtime-info.json" 0 runtime_incomplete
  IfFileExists "$INSTDIR\quantem-server\*.*" 0 runtime_promote
    ClearErrors
    Rename "$INSTDIR\quantem-server" "$QRuntimeBackup"
    ${If} ${Errors}
      Abort "The existing QuantEM runtime could not be replaced."
    ${EndIf}
  runtime_promote:
  ClearErrors
  Rename "$QRuntimeStage\quantem-server" "$INSTDIR\quantem-server"
  ${If} ${Errors}
    Rename "$QRuntimeBackup" "$INSTDIR\quantem-server"
    Abort "The new QuantEM runtime could not be activated; the previous runtime was restored."
  ${EndIf}
  StrCpy $QRuntimeChanged 1
  DetailPrint "Installed new $QRuntimeVariant runtime $QRuntimeRequiredId."
  Goto runtime_ready
  runtime_incomplete:
    Abort "The verified runtime archive is incomplete."

  runtime_ready:
  ${If} $QRuntimeChanged <> 1
    Call QuantemBackupApplication
  ${EndIf}
  Call QuantemPromoteApplication
  ${If} $R7 <> 0
    ${If} $QRuntimeChanged = 1
      RMDir /r "$INSTDIR\quantem-server"
      Rename "$QRuntimeBackup" "$INSTDIR\quantem-server"
    ${Else}
      Call QuantemRestoreApplication
    ${EndIf}
    Abort "The QuantEM application update failed; the previous version was restored."
  ${EndIf}
  RMDir /r "$QApplicationBackup"
  RMDir /r "$QRuntimeBackup"
  RMDir /r "$INSTDIR\.quantem-install"
  DetailPrint "Installed QuantEM application ${QPAYLOAD_VERSION}."
FunctionEnd

!macroend

!macro QUANTEM_MODEL_PAGE_FUNCTIONS

; Read all eight checkbox states into the state variables.
Function QuantemReadPackStates
  ${NSD_GetState} $QMdlCbOmniMito $QMdlStOmniMito
  ${NSD_GetState} $QMdlCbOmniEr $QMdlStOmniEr
  ${NSD_GetState} $QMdlCbOmniNuc $QMdlStOmniNuc
  ${NSD_GetState} $QMdlCbOmniLd $QMdlStOmniLd
  ${NSD_GetState} $QMdlCbQemMito $QMdlStQemMito
  ${NSD_GetState} $QMdlCbQemEr $QMdlStQemEr
  ${NSD_GetState} $QMdlCbQemNuc $QMdlStQemNuc
  ${NSD_GetState} $QMdlCbQemLd $QMdlStQemLd
FunctionEnd

; Recompute the honest download total: selected head sizes, plus each
; family's shared trunk exactly once if any pack of that family is selected.
Function QuantemRecomputeTotal
  Call QuantemReadPackStates
  StrCpy $0 0   ; running total, decimal MB
  StrCpy $1 0   ; any OmniEM pack selected
  StrCpy $2 0   ; any QuantEM pack selected
  ${If} $QMdlStOmniMito = ${BST_CHECKED}
    IntOp $0 $0 + 26
    StrCpy $1 1
  ${EndIf}
  ${If} $QMdlStOmniEr = ${BST_CHECKED}
    IntOp $0 $0 + 135
    StrCpy $1 1
  ${EndIf}
  ${If} $QMdlStOmniNuc = ${BST_CHECKED}
    IntOp $0 $0 + 26
    StrCpy $1 1
  ${EndIf}
  ${If} $QMdlStOmniLd = ${BST_CHECKED}
    IntOp $0 $0 + 26
    StrCpy $1 1
  ${EndIf}
  ${If} $QMdlStQemMito = ${BST_CHECKED}
    IntOp $0 $0 + 137
    StrCpy $2 1
  ${EndIf}
  ${If} $QMdlStQemEr = ${BST_CHECKED}
    IntOp $0 $0 + 465
    StrCpy $2 1
  ${EndIf}
  ${If} $QMdlStQemNuc = ${BST_CHECKED}
    IntOp $0 $0 + 137
    StrCpy $2 1
  ${EndIf}
  ${If} $QMdlStQemLd = ${BST_CHECKED}
    IntOp $0 $0 + 137
    StrCpy $2 1
  ${EndIf}
  ${If} $1 = 1
    IntOp $0 $0 + 1218    ; shared OmniEM encoder, once
  ${EndIf}
  ${If} $2 = 1
    IntOp $0 $0 + 228     ; shared QuantEM encoder, once
  ${EndIf}
  ${If} $0 = 0
    ${NSD_SetText} $QMdlTotalLabel "No model packs will be downloaded on first launch."
  ${ElseIf} $0 < 1000
    ${NSD_SetText} $QMdlTotalLabel "Estimated download on first launch: $0 MB"
  ${Else}
    IntOp $3 $0 / 1000
    IntOp $4 $0 % 1000
    IntOp $4 $4 / 10
    ${If} $4 < 10
      ${NSD_SetText} $QMdlTotalLabel "Estimated download on first launch: $3.0$4 GB"
    ${Else}
      ${NSD_SetText} $QMdlTotalLabel "Estimated download on first launch: $3.$4 GB"
    ${EndIf}
  ${EndIf}
FunctionEnd

; Click callback shared by all eight checkboxes (stack carries the hwnd).
Function QuantemOnPackClick
  Pop $0
  Call QuantemRecomputeTotal
FunctionEnd

Function PageQuantemModels
  ; Passive and update installs never show this page (and QMdlConfigured
  ; stays 0, so the POSTINSTALL hook leaves any existing selection alone).
  ${If} $PassiveMode = 1
  ${OrIf} $UpdateMode = 1
    Abort
  ${EndIf}

  !insertmacro MUI_HEADER_TEXT "Model downloads" "Choose the segmentation models QuantEM downloads on first launch."

  nsDialogs::Create 1018
  Pop $QMdlDialog
  ${If} $QMdlDialog == error
    Abort
  ${EndIf}

  ${NSD_CreateLabel} 0 0 100% 16u "Selected packs are downloaded and verified when QuantEM first starts (progress and cancel in the app). Packs can be added or removed at any time on the Models screen."
  Pop $0

  ${NSD_CreateLabel} 0 20u 100% 16u "OmniEM packs (recommended) - the first OmniEM pack also downloads the shared OmniEM encoder (1.22 GB):"
  Pop $0
  ${NSD_CreateCheckbox} 0 38u 49% 10u "Mitochondria (26 MB)"
  Pop $QMdlCbOmniMito
  ${NSD_CreateCheckbox} 50% 38u 49% 10u "Endoplasmic reticulum (135 MB)"
  Pop $QMdlCbOmniEr
  ${NSD_CreateCheckbox} 0 49u 49% 10u "Nucleus (26 MB)"
  Pop $QMdlCbOmniNuc
  ${NSD_CreateCheckbox} 50% 49u 49% 10u "Lipid droplets (26 MB)"
  Pop $QMdlCbOmniLd

  ${NSD_CreateLabel} 0 63u 100% 16u "QuantEM packs - the first QuantEM pack also downloads the shared QuantEM encoder (228 MB):"
  Pop $0
  ${NSD_CreateCheckbox} 0 81u 49% 10u "Mitochondria (137 MB)"
  Pop $QMdlCbQemMito
  ${NSD_CreateCheckbox} 50% 81u 49% 10u "Endoplasmic reticulum (465 MB)"
  Pop $QMdlCbQemEr
  ${NSD_CreateCheckbox} 0 92u 49% 10u "Nucleus (137 MB)"
  Pop $QMdlCbQemNuc
  ${NSD_CreateCheckbox} 50% 92u 49% 10u "Lipid droplets (137 MB)"
  Pop $QMdlCbQemLd

  ${NSD_CreateLabel} 0 110u 100% 12u ""
  Pop $QMdlTotalLabel

  ${If} $QMdlConfigured = 1
    ; Returning to the page (Back from a later page): restore the selection.
    ${NSD_SetState} $QMdlCbOmniMito $QMdlStOmniMito
    ${NSD_SetState} $QMdlCbOmniEr $QMdlStOmniEr
    ${NSD_SetState} $QMdlCbOmniNuc $QMdlStOmniNuc
    ${NSD_SetState} $QMdlCbOmniLd $QMdlStOmniLd
    ${NSD_SetState} $QMdlCbQemMito $QMdlStQemMito
    ${NSD_SetState} $QMdlCbQemEr $QMdlStQemEr
    ${NSD_SetState} $QMdlCbQemNuc $QMdlStQemNuc
    ${NSD_SetState} $QMdlCbQemLd $QMdlStQemLd
  ${Else}
    ; First show: the four OmniEM packs checked, the four QuantEM unchecked.
    ${NSD_Check} $QMdlCbOmniMito
    ${NSD_Check} $QMdlCbOmniEr
    ${NSD_Check} $QMdlCbOmniNuc
    ${NSD_Check} $QMdlCbOmniLd
  ${EndIf}

  ${NSD_OnClick} $QMdlCbOmniMito QuantemOnPackClick
  ${NSD_OnClick} $QMdlCbOmniEr QuantemOnPackClick
  ${NSD_OnClick} $QMdlCbOmniNuc QuantemOnPackClick
  ${NSD_OnClick} $QMdlCbOmniLd QuantemOnPackClick
  ${NSD_OnClick} $QMdlCbQemMito QuantemOnPackClick
  ${NSD_OnClick} $QMdlCbQemEr QuantemOnPackClick
  ${NSD_OnClick} $QMdlCbQemNuc QuantemOnPackClick
  ${NSD_OnClick} $QMdlCbQemLd QuantemOnPackClick

  Call QuantemRecomputeTotal
  nsDialogs::Show
FunctionEnd

Function PageLeaveQuantemModels
  Call QuantemReadPackStates
  StrCpy $QMdlConfigured 1
FunctionEnd

; Write <INSTDIR>\data\pending-model-installs.json from the recorded
; selection. Called from NSIS_HOOK_POSTINSTALL (after all files are in
; place). Written via a temp file + rename so the app can never observe a
; half-written selection.
Function QuantemWritePendingInstalls
  ${If} $QMdlConfigured <> 1
    Return
  ${EndIf}
  StrCpy $R8 ""
  ${If} $QMdlStOmniMito = ${BST_CHECKED}
    StrCpy $R8 '$R8, "omniem:mito"'
  ${EndIf}
  ${If} $QMdlStOmniEr = ${BST_CHECKED}
    StrCpy $R8 '$R8, "omniem:er"'
  ${EndIf}
  ${If} $QMdlStOmniNuc = ${BST_CHECKED}
    StrCpy $R8 '$R8, "omniem:nucleus"'
  ${EndIf}
  ${If} $QMdlStOmniLd = ${BST_CHECKED}
    StrCpy $R8 '$R8, "omniem:ld"'
  ${EndIf}
  ${If} $QMdlStQemMito = ${BST_CHECKED}
    StrCpy $R8 '$R8, "quantem:mito"'
  ${EndIf}
  ${If} $QMdlStQemEr = ${BST_CHECKED}
    StrCpy $R8 '$R8, "quantem:er"'
  ${EndIf}
  ${If} $QMdlStQemNuc = ${BST_CHECKED}
    StrCpy $R8 '$R8, "quantem:nucleus"'
  ${EndIf}
  ${If} $QMdlStQemLd = ${BST_CHECKED}
    StrCpy $R8 '$R8, "quantem:ld"'
  ${EndIf}
  ${If} $R8 == ""
    ; The user explicitly chose nothing: remove any stale selection from a
    ; previous install into the same directory.
    Delete "$INSTDIR\data\pending-model-installs.json"
    Return
  ${EndIf}
  StrCpy $R8 $R8 "" 2   ; drop the leading ", "
  CreateDirectory "$INSTDIR\data"
  ClearErrors
  FileOpen $R9 "$INSTDIR\data\pending-model-installs.json.tmp" w
  ${If} ${Errors}
    DetailPrint "Could not record the model selection; models can be installed from the Models screen."
    Return
  ${EndIf}
  FileWrite $R9 '{"packs": [$R8]}'
  FileClose $R9
  Delete "$INSTDIR\data\pending-model-installs.json"
  Rename "$INSTDIR\data\pending-model-installs.json.tmp" "$INSTDIR\data\pending-model-installs.json"
  DetailPrint "Recorded model selection for first launch: data\pending-model-installs.json"
FunctionEnd

!macroend

!macro NSIS_HOOK_POSTINSTALL
  WriteRegStr SHCTX "${MANUPRODUCTKEY}" "RuntimeVariant" "$QRuntimeVariant"
  WriteRegStr SHCTX "${MANUPRODUCTKEY}" "RuntimeVersion" "${QPAYLOAD_VERSION}"
  WriteRegStr SHCTX "${MANUPRODUCTKEY}" "RuntimeId" "$QRuntimeRequiredId"
  FileOpen $0 "$INSTDIR\.quantem-runtime-variant.tmp" w
  FileWrite $0 "$QRuntimeVariant"
  FileClose $0
  Delete "$INSTDIR\.quantem-runtime-variant"
  Rename "$INSTDIR\.quantem-runtime-variant.tmp" "$INSTDIR\.quantem-runtime-variant"
  FileOpen $0 "$INSTDIR\.quantem-runtime-id.tmp" w
  FileWrite $0 "$QRuntimeRequiredId"
  FileClose $0
  Delete "$INSTDIR\.quantem-runtime-id"
  Rename "$INSTDIR\.quantem-runtime-id.tmp" "$INSTDIR\.quantem-runtime-id"
  Call QuantemWritePendingInstalls
!macroend

!macro NSIS_HOOK_PREINSTALL
  Call QuantemInstallRuntime
!macroend

; The app's data directory lives at <INSTDIR>\data (DB, models, HF cache,
; exports, logs, WebView2 profile). The stock uninstaller does not know about
; it: without this hook a plain "RMDir $INSTDIR" leaves the whole tree behind
; even when the user asked for their data to be deleted.
!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $UpdateMode <> 1
    Delete "$INSTDIR\.quantem-runtime-variant"
    Delete "$INSTDIR\.quantem-runtime-id"
    RMDir /r "$INSTDIR\quantem-server"
    RMDir /r "$INSTDIR\.quantem-install"
    ; The pending file is installer metadata, not user data.
    Delete "$INSTDIR\data\pending-model-installs.json"
    ${If} $DeleteAppDataCheckboxState = 1
      RMDir /r "$INSTDIR\data"
    ${EndIf}
    ; Remove the directories only if empty (data survives unless opted in).
    RMDir "$INSTDIR\data"
    RMDir "$INSTDIR"
  ${EndIf}
!macroend
