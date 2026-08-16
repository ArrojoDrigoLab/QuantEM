import { afterEach, describe, expect, it, vi } from "vitest";
import { invoke } from "@tauri-apps/api/core";
import {
  saveTextFile,
  saveUrlFile,
  toCsv,
  toCsvField,
} from "@/utils/downloadText";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

const tauriGlobal = globalThis as typeof globalThis & {
  isTauri?: boolean;
  __TAURI_INTERNALS__?: unknown;
};

afterEach(() => {
  delete window.showSaveFilePicker;
  delete tauriGlobal.isTauri;
  delete tauriGlobal.__TAURI_INTERNALS__;
  vi.clearAllMocks();
  vi.unstubAllGlobals();
});

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

describe("native save picker", () => {
  it("uses the Tauri save command for generated text in desktop builds", async () => {
    tauriGlobal.isTauri = true;
    vi.mocked(invoke).mockResolvedValue(true);
    window.showSaveFilePicker = vi.fn();

    await expect(
      saveTextFile("distance-bands-run-1.csv", "band,count\n0-50,2\n", "text/csv")
    ).resolves.toBe("saved");

    expect(invoke).toHaveBeenCalledWith("save_text_file", {
      request: {
        suggestedName: "distance-bands-run-1.csv",
        mimeType: "text/csv",
        contents: "band,count\n0-50,2\n",
      },
    });
    expect(window.showSaveFilePicker).not.toHaveBeenCalled();
  });

  it("streams server exports through the Tauri shell in desktop builds", async () => {
    tauriGlobal.__TAURI_INTERNALS__ = {};
    vi.mocked(invoke).mockResolvedValue(false);
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      saveUrlFile(
        "Portal_field_EM_8bit.png",
        "http://127.0.0.1:8722/api/assets/asset-1/export-png/?source=original",
        "image/png"
      )
    ).resolves.toBe("cancelled");

    expect(invoke).toHaveBeenCalledWith("save_url_file", {
      request: {
        suggestedName: "Portal_field_EM_8bit.png",
        mimeType: "image/png",
        url: "http://127.0.0.1:8722/api/assets/asset-1/export-png/?source=original",
      },
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("turns a native command rejection into a user-facing Error", async () => {
    tauriGlobal.isTauri = true;
    vi.mocked(invoke).mockRejectedValue("The selected folder is read-only.");

    await expect(saveTextFile("objects.csv", "id\n1\n")).rejects.toThrow(
      "The selected folder is read-only."
    );
  });

  it("offers the default CSV name and writes generated text to the selected file", async () => {
    const write = vi.fn(async (value: Blob) => {
      void value;
    });
    const close = vi.fn(async () => undefined);
    const abort = vi.fn(async () => undefined);
    const createWritable = vi.fn(async () => ({ write, close, abort }));
    const picker = vi.fn(
      async () => ({ createWritable }) as unknown as FileSystemFileHandle
    );
    window.showSaveFilePicker = picker;

    await expect(
      saveTextFile("distance-bands-run-1.csv", "band,count\n0-50,2\n", "text/csv")
    ).resolves.toBe("saved");

    expect(picker).toHaveBeenCalledWith({
      suggestedName: "distance-bands-run-1.csv",
      types: [
        {
          description: "CSV file",
          accept: { "text/csv": [".csv"] },
        },
      ],
    });
    expect(createWritable).toHaveBeenCalledTimes(1);
    expect(write).toHaveBeenCalledTimes(1);
    expect(write.mock.calls[0][0]).toBeInstanceOf(Blob);
    expect(close).toHaveBeenCalledTimes(1);
    expect(abort).not.toHaveBeenCalled();
  });

  it("streams a server export into the chosen file", async () => {
    const chunks: Uint8Array[] = [];
    const writable = new WritableStream<Uint8Array>({
      write(chunk) {
        chunks.push(chunk);
      },
    });
    const createWritable = vi.fn(async () => writable as FileSystemWritableFileStream);
    window.showSaveFilePicker = vi.fn(
      async () => ({ createWritable }) as unknown as FileSystemFileHandle
    );
    const fetchMock = vi.fn(async () =>
      new Response(new Uint8Array([137, 80, 78, 71]), {
        status: 200,
        headers: { "content-type": "image/png" },
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      saveUrlFile("Portal_field_EM_8bit.png", "/api/export-png", "image/png")
    ).resolves.toBe("saved");

    expect(fetchMock).toHaveBeenCalledWith("/api/export-png", {
      credentials: "same-origin",
    });
    expect(Array.from(chunks[0])).toEqual([137, 80, 78, 71]);
  });

  it("treats closing the Save dialog as a cancellation without requesting the export", async () => {
    window.showSaveFilePicker = vi.fn(async () => {
      throw new DOMException("The user cancelled.", "AbortError");
    });
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    await expect(
      saveUrlFile("objects.csv", "/api/objects.csv", "text/csv")
    ).resolves.toBe("cancelled");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
