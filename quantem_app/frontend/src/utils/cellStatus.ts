import type {
  CellStatus,
  CellStatusLabel,
  LabelState,
  RefinementStatus,
} from "@/shared/types/common";

export const CELL_STATUS_CANDIDATE = 0;
export const CELL_STATUS_INITIAL_CONFIRM = 1;
export const CELL_STATUS_MODEL_OUTPUT_GEOMETRY = 2;
export const CELL_STATUS_REFINED = 10;

export const CELL_SELECTABLE_STATUSES: CellStatus[] = [
  CELL_STATUS_CANDIDATE,
  CELL_STATUS_INITIAL_CONFIRM,
  CELL_STATUS_MODEL_OUTPUT_GEOMETRY,
  CELL_STATUS_REFINED,
];

type CellStatusSource = {
  status?: CellStatus | number | null;
  label_state?: LabelState | null;
  refined?: RefinementStatus | null;
};

export function normalizeCellStatus(value: unknown): CellStatus | null {
  const numericValue = typeof value === "string" ? Number(value) : value;
  if (
    numericValue === CELL_STATUS_CANDIDATE ||
    numericValue === CELL_STATUS_INITIAL_CONFIRM ||
    numericValue === CELL_STATUS_MODEL_OUTPUT_GEOMETRY ||
    numericValue === CELL_STATUS_REFINED
  ) {
    return numericValue;
  }
  return null;
}

export function cellStatusLabel(status: CellStatus): CellStatusLabel {
  if (status === CELL_STATUS_REFINED) {
    return "REFINED";
  }
  if (status === CELL_STATUS_MODEL_OUTPUT_GEOMETRY) {
    return "MODEL_OUTPUT_GEOMETRY";
  }
  if (status === CELL_STATUS_INITIAL_CONFIRM) {
    return "INITIAL_CONFIRM";
  }
  return "CANDIDATE";
}

export function cellStatusFromLegacy(
  labelState: LabelState | null | undefined,
  refined: RefinementStatus | null | undefined
): CellStatus {
  if (labelState === "CONFIRMED" || (!labelState && refined)) {
    if (refined === "AUTOMATIC") {
      return CELL_STATUS_MODEL_OUTPUT_GEOMETRY;
    }
    return refined === "MANUAL" ? CELL_STATUS_REFINED : CELL_STATUS_INITIAL_CONFIRM;
  }
  return CELL_STATUS_CANDIDATE;
}

export function getCellStatus(source: CellStatusSource): CellStatus {
  return (
    normalizeCellStatus(source.status) ??
    cellStatusFromLegacy(source.label_state, source.refined)
  );
}

export function getCellStatusLabel(source: CellStatusSource): CellStatusLabel {
  return cellStatusLabel(getCellStatus(source));
}

export function isCellCandidateStatus(status: CellStatus | null | undefined): boolean {
  return status === CELL_STATUS_CANDIDATE;
}

export function isCellInitialConfirmStatus(
  status: CellStatus | null | undefined
): boolean {
  return status === CELL_STATUS_INITIAL_CONFIRM;
}

export function isCellRefinedStatus(status: CellStatus | null | undefined): boolean {
  return status === CELL_STATUS_REFINED;
}

export function isCellModelOutputGeometryStatus(
  status: CellStatus | null | undefined
): boolean {
  return status === CELL_STATUS_MODEL_OUTPUT_GEOMETRY;
}

export function isCellConfirmedStatus(status: CellStatus | null | undefined): boolean {
  return (
    status === CELL_STATUS_INITIAL_CONFIRM ||
    status === CELL_STATUS_MODEL_OUTPUT_GEOMETRY ||
    status === CELL_STATUS_REFINED
  );
}
