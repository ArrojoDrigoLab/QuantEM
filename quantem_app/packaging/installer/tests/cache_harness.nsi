Unicode true
!include MUI2.nsh
!include nsDialogs.nsh
!include "${CACHE_HOOK}"
Name "QuantEM cache test"
OutFile "${TEST_ROOT}\cache-harness.exe"
RequestExecutionLevel user
SilentInstall silent
Var PassiveMode
Var UpdateMode
!insertmacro QUANTEM_DATABASE_CACHE_PAGE_FUNCTIONS
Page custom PageQuantemDatabaseCache PageLeaveQuantemDatabaseCache
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"

!macro CheckPolicy RAM
  StrCpy $QCacheRamMB ${RAM}
  Call QuantemCachePolicy
  FileWrite $9 '{"ram_mb":${RAM},"recommended":$QCacheRecommendedMB,"maximum":$QCacheMaxMB}$\r$\n'
!macroend

!macro CheckValue VALUE
  StrCpy $0 "${VALUE}"
  Call QuantemValidateDatabaseCache
  ${If} ${Errors}
    StrCpy $2 "false"
  ${Else}
    StrCpy $2 "true"
  ${EndIf}
  FileWrite $9 '{"input":"${VALUE}","valid":$2}$\r$\n'
!macroend

Section
  StrCpy $UpdateMode 0
  StrCpy $PassiveMode 1
  CreateDirectory "$EXEDIR\nsis-tests"
  FileOpen $9 "$EXEDIR\nsis-tests\results.jsonl" w
  Call QuantemInitializeDatabaseCache
  FileWrite $9 '{"detected_ram_mb":$QCacheRamMB,"suggestion":$QCacheMB}$\r$\n'
  !insertmacro CheckPolicy 0
  !insertmacro CheckPolicy 4096
  !insertmacro CheckPolicy 8192
  !insertmacro CheckPolicy 16384
  !insertmacro CheckPolicy 32768
  !insertmacro CheckPolicy 65536
  !insertmacro CheckPolicy 262144
  StrCpy $QCacheMaxMB 4096
  !insertmacro CheckValue "32"
  !insertmacro CheckValue "1024"
  !insertmacro CheckValue "4096"
  !insertmacro CheckValue "4097"
  !insertmacro CheckValue "31"
  !insertmacro CheckValue ""
  !insertmacro CheckValue "123junk"
  !insertmacro CheckValue "256.5"
  !insertmacro CheckValue "4294968320"
  FileClose $9

  StrCpy $INSTDIR "$EXEDIR\nsis-tests\fresh"
  StrCpy $QCacheMB 768
  Call QuantemWriteDatabaseCache

  StrCpy $INSTDIR "$EXEDIR\nsis-tests\existing-settings"
  CreateDirectory "$INSTDIR\data"
  FileOpen $0 "$INSTDIR\data\settings.json" w
  FileWrite $0 '{"database_cache_mb":1024,"pre_migration_backup_retention_count":3}'
  FileClose $0
  Call QuantemWriteDatabaseCache

  StrCpy $INSTDIR "$EXEDIR\nsis-tests\existing-postgres"
  CreateDirectory "$INSTDIR\data\postgres\cluster"
  FileOpen $0 "$INSTDIR\data\postgres\cluster\PG_VERSION" w
  FileWrite $0 '17'
  FileClose $0
  FileOpen $0 "$INSTDIR\data\postgres\cluster\postgresql.auto.conf" w
  FileWrite $0 "shared_buffers = '1GB'$\r$\n"
  FileClose $0
  Call QuantemWriteDatabaseCache

  StrCpy $INSTDIR "$EXEDIR\nsis-tests\legacy-sqlite"
  CreateDirectory "$INSTDIR\data"
  FileOpen $0 "$INSTDIR\data\quantem.sqlite3" w
  FileWrite $0 "legacy database sentinel"
  FileClose $0
  Call QuantemWriteDatabaseCache

  StrCpy $UpdateMode 1
  StrCpy $INSTDIR "$EXEDIR\nsis-tests\upgrade"
  Call QuantemWriteDatabaseCache

  StrCpy $UpdateMode 0
  StrCpy $QCacheInitialized 0
  StrCpy $INSTDIR "$EXEDIR\nsis-tests\silent-fresh"
  Call QuantemWriteDatabaseCache
SectionEnd
