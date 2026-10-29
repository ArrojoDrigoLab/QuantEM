; Installation-time database RAM choice. Upgrades leave existing data settings
; and PostgreSQL configuration untouched; later edits live in app Settings.
Var QCacheDialog
Var QCacheInput
Var QCacheInitialized
Var QCacheMB
Var QCacheRamMB
Var QCacheRecommendedMB
Var QCacheMaxMB

!macro QUANTEM_DATABASE_CACHE_PAGE_FUNCTIONS

Function QuantemCachePolicy
  ; Keep in sync with core/database_cache.py::cache_limits.
  StrCpy $QCacheRecommendedMB 256
  StrCpy $QCacheMaxMB 4096
  ${If} $QCacheRamMB > 0
    IntOp $QCacheMaxMB $QCacheRamMB / 4
    ${If} $QCacheMaxMB > 65536
      StrCpy $QCacheMaxMB 65536
    ${ElseIf} $QCacheMaxMB < 32
      StrCpy $QCacheMaxMB 32
    ${EndIf}
    IntOp $QCacheRecommendedMB $QCacheRamMB / 32
    IntOp $QCacheRecommendedMB $QCacheRecommendedMB / 64
    IntOp $QCacheRecommendedMB $QCacheRecommendedMB * 64
    ${If} $QCacheRecommendedMB < 128
      StrCpy $QCacheRecommendedMB 128
    ${ElseIf} $QCacheRecommendedMB > 4096
      StrCpy $QCacheRecommendedMB 4096
    ${EndIf}
    ${If} $QCacheRecommendedMB > $QCacheMaxMB
      StrCpy $QCacheRecommendedMB $QCacheMaxMB
    ${EndIf}
  ${EndIf}
FunctionEnd

Function QuantemInitializeDatabaseCache
  ${If} $QCacheInitialized = 1
    Return
  ${EndIf}
  StrCpy $QCacheRamMB 0
  ; MEMORYSTATUSEX uses 64-bit byte counts, even in the 32-bit installer.
  System::Alloc 64
  Pop $0
  ${If} $0 <> 0
    System::Call '*$0(i 64)'
    System::Call 'kernel32::GlobalMemoryStatusEx(p r0) i .r1'
    ${If} $1 <> 0
      System::Call '*$0(i, i, l .r2)'
      System::Int64Op $2 / 1048576
      Pop $QCacheRamMB
    ${EndIf}
    System::Free $0
  ${EndIf}
  Call QuantemCachePolicy
  StrCpy $QCacheMB $QCacheRecommendedMB
  StrCpy $QCacheInitialized 1
FunctionEnd

; Returning with the error flag means an existing installation must be kept.
Function QuantemCacheIsExistingInstall
  ClearErrors
  ${If} $UpdateMode = 1
  ${OrIf} ${FileExists} "$INSTDIR\data\settings.json"
  ${OrIf} ${FileExists} "$INSTDIR\data\postgres\cluster\PG_VERSION"
  ${OrIf} ${FileExists} "$INSTDIR\data\quantem.sqlite3"
    SetErrors
  ${EndIf}
FunctionEnd

Function PageQuantemDatabaseCache
  Call QuantemCacheIsExistingInstall
  ${If} ${Errors}
    Abort
  ${EndIf}
  ${If} $PassiveMode = 1
    Abort
  ${EndIf}
  Call QuantemInitializeDatabaseCache
  !insertmacro MUI_HEADER_TEXT "Database memory" "Choose how much RAM QuantEM uses to cache database pages."
  nsDialogs::Create 1018
  Pop $QCacheDialog
  ${If} $QCacheDialog == error
    Abort
  ${EndIf}
  ${NSD_CreateLabel} 0 0 100% 30u "The cache uses RAM while QuantEM is open. Image processing and other applications need additional memory. This does not set the size of files stored on disk."
  Pop $0
  ${If} $QCacheRamMB > 0
    StrCpy $0 "Detected RAM: $QCacheRamMB MiB. Suggested cache: $QCacheRecommendedMB MiB."
  ${Else}
    StrCpy $0 "RAM could not be detected. Suggested cache: $QCacheRecommendedMB MiB."
  ${EndIf}
  ${NSD_CreateLabel} 0 42u 100% 24u $0
  Pop $0
  ${NSD_CreateLabel} 0 79u 125u 12u "Database cache (MiB):"
  Pop $0
  ${NSD_CreateNumber} 130u 76u 75u 14u $QCacheMB
  Pop $QCacheInput
  ${NSD_CreateLabel} 0 102u 100% 40u "Enter a whole number from 32 to $QCacheMaxMB MiB. You can change this in Settings later. Upgrades keep your current value."
  Pop $0
  nsDialogs::Show
FunctionEnd

Function QuantemValidateDatabaseCache
  ClearErrors
  IntOp $1 $0 + 0
  ${If} $0 != $1
  ${OrIf} $1 < 32
  ${OrIf} $1 > $QCacheMaxMB
    SetErrors
  ${EndIf}
FunctionEnd

Function PageLeaveQuantemDatabaseCache
  ${NSD_GetText} $QCacheInput $0
  Call QuantemValidateDatabaseCache
  ${If} ${Errors}
    MessageBox MB_OK|MB_ICONEXCLAMATION "Enter a whole number from 32 to $QCacheMaxMB MiB."
    Abort
  ${EndIf}
  StrCpy $QCacheMB $1
FunctionEnd

Function QuantemWriteDatabaseCache
  Call QuantemCacheIsExistingInstall
  ${If} ${Errors}
    DetailPrint "Preserving the existing database cache setting."
    Return
  ${EndIf}
  ; Silent fresh installations use the recommendation without a wizard.
  Call QuantemInitializeDatabaseCache
  CreateDirectory "$INSTDIR\data"
  ClearErrors
  FileOpen $0 "$INSTDIR\data\settings.json.install.tmp" w
  ${IfNot} ${Errors}
    FileWrite $0 '{"schema_version": 1, "database_cache_mb": $QCacheMB}'
    FileClose $0
    ${IfNot} ${Errors}
      ; Rename fails if another process created settings.json in the meantime.
      Rename "$INSTDIR\data\settings.json.install.tmp" "$INSTDIR\data\settings.json"
    ${EndIf}
  ${EndIf}
  ${If} ${Errors}
    DetailPrint "Could not save the database cache choice; set it in QuantEM Settings."
    Delete "$INSTDIR\data\settings.json.install.tmp"
  ${Else}
    DetailPrint "Database memory cache: $QCacheMB MiB."
  ${EndIf}
FunctionEnd

!macroend
