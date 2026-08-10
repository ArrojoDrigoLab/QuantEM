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
  Call QuantemWritePendingInstalls
!macroend

; The app's data directory lives at <INSTDIR>\data (DB, models, HF cache,
; exports, logs, WebView2 profile). The stock uninstaller does not know about
; it: without this hook a plain "RMDir $INSTDIR" leaves the whole tree behind
; even when the user asked for their data to be deleted.
!macro NSIS_HOOK_POSTUNINSTALL
  ${If} $UpdateMode <> 1
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
