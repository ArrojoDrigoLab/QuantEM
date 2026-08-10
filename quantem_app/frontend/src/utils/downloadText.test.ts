import { describe, expect, it } from "vitest";
import { toCsv, toCsvField } from "@/utils/downloadText";

describe("toCsvField", () => {
  it("leaves plain values alone", () => {
    expect(toCsvField("mito")).toBe("mito");
    expect(toCsvField(0.5)).toBe("0.5");
  });

  it("renders null and undefined as an empty field, not the string 'null'", () => {
    expect(toCsvField(null)).toBe("");
    expect(toCsvField(undefined)).toBe("");
  });

  it("quotes and escapes anything that would break the row", () => {
    expect(toCsvField('a,b')).toBe('"a,b"');
    expect(toCsvField('say "hi"')).toBe('"say ""hi"""');
    expect(toCsvField("line\nbreak")).toBe('"line\nbreak"');
  });
});

describe("toCsv", () => {
  it("writes a header row and one row per record", () => {
    expect(
      toCsv(
        ["band", "count"],
        [
          ["0–50 nm", 12],
          ["50–100 nm", 0],
        ]
      )
    ).toBe("band,count\n0–50 nm,12\n50–100 nm,0\n");
  });
});
