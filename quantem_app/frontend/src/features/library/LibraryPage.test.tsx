import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LibraryPage } from "@/features/library/LibraryPage";
import { server } from "@/test/msw/server";
import { getAsset, getHomeEntryPage, uploadAsset } from "@/shared/api/assets";
import { getSystemStatus } from "@/shared/api/jobs";
import type {
  AssetDetail,
  HomeEntry,
  HomeEntryPage,
} from "@/shared/types/images";

const navigateMock = vi.fn();

vi.mock("react-router-dom", async () => {
  const actual =
    await vi.importActual<typeof import("react-router-dom")>("react-router-dom");
  return { ...actual, useNavigate: () => navigateMock };
});

vi.mock("@/shared/api/assets", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/assets")>(
      "@/shared/api/assets"
    );
  return {
    ...actual,
    getHomeEntryPage: vi.fn(),
    getAsset: vi.fn(),
    deleteAsset: vi.fn(),
    uploadAsset: vi.fn(),
  };
});

vi.mock("@/shared/api/jobs", async () => {
  const actual =
    await vi.importActual<typeof import("@/shared/api/jobs")>("@/shared/api/jobs");
  return { ...actual, getSystemStatus: vi.fn(), getJobQueueStatus: vi.fn() };
});

const EMPTY_PAGE: HomeEntryPage = {
  results: [],
  total: 0,
  limit: 60,
  offset: 0,
  has_more: false,
};

function makeEntry(overrides: Partial<HomeEntry> = {}): HomeEntry {
  return {
    id: "asset-existing",
    display_name: "Liver 01",
    original_filename: "liver01.tif",
    metadata_summary: "1024x1024",
    width: 1024,
    height: 1024,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    preprocess_stage: "DONE",
    preprocess_progress: 100,
    ngff_ready: true,
    can_open: true,
    ...overrides,
  };
}

function pageOf(entries: HomeEntry[]): HomeEntryPage {
  return {
    results: entries,
    total: entries.length,
    limit: 60,
    offset: 0,
    has_more: false,
  };
}

function makeUploadedAsset(overrides: Partial<AssetDetail> = {}): AssetDetail {
  return {
    id: "asset-new",
    file_path: "",
    original_filename: "grid2.tif",
    display_name: "grid2",
    is_eval_set: false,
    width: 8192,
    height: 8192,
    channels: 1,
    bit_depth: 8,
    pixel_size_nm: 5,
    preprocess_stage: "ENCODING",
    preprocess_progress: 0,
    ngff_ready: false,
    is_workable: true,
    tags: [],
    created_at: "2026-02-02T00:00:00Z",
    updated_at: "2026-02-02T00:00:00Z",
    ...overrides,
  };
}

function renderLibrary() {
  return render(
    <MemoryRouter>
      <LibraryPage />
    </MemoryRouter>
  );
}

/** Drop a TIFF on the page and wait for the confirmation strip. */
async function importOneFile() {
  const zone = await screen.findByTestId("import-drop-zone");
  fireEvent.drop(zone, {
    dataTransfer: {
      files: [new File([new Uint8Array(16)], "grid2.tif", { type: "image/tiff" })],
      types: ["Files"],
    },
  });
  await screen.findByTestId("import-chosen-file");
  await act(async () => {
    fireEvent.submit(screen.getByTestId("import-form"));
  });
  return screen.findByTestId("import-confirmation");
}

/** Drop several TIFFs on the page and import all of them. */
async function importFiles(names: string[]) {
  const zone = await screen.findByTestId("import-drop-zone");
  fireEvent.drop(zone, {
    dataTransfer: {
      files: names.map(
        (name) => new File([new Uint8Array(16)], name, { type: "image/tiff" })
      ),
      types: ["Files"],
    },
  });
  await waitFor(() =>
    expect(screen.getAllByTestId("import-file-row")).toHaveLength(names.length)
  );
  await act(async () => {
    fireEvent.submit(screen.getByTestId("import-form"));
  });
  return screen.findByTestId("import-confirmation");
}

describe("LibraryPage", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
    window.localStorage.setItem("quantem-workflow-guide-dismissed-v1", "1");
    vi.mocked(getHomeEntryPage).mockResolvedValue(EMPTY_PAGE);
    vi.mocked(getSystemStatus).mockResolvedValue({
      cuda_available: false,
      supported_upload_formats: [".tif", ".tiff", ".png"],
    });
    vi.mocked(uploadAsset).mockResolvedValue(makeUploadedAsset());
    vi.mocked(getAsset).mockResolvedValue(makeUploadedAsset());
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("states CPU-only as a fact, not an amber warning", async () => {
    renderLibrary();

    // The README calls CPU-only fully supported and every released model runs
    // on it; an amber "CUDA unavailable" badge on the first screen reads as a
    // broken install.
    const badge = await screen.findByText("Running on CPU");
    expect(badge.className).not.toContain("amber");
    expect(badge).toHaveAttribute("title", expect.stringContaining("Everything works on CPU"));
    expect(screen.queryByText(/cuda unavailable/i)).toBeNull();
  });

  it("badges a real GPU as good news", async () => {
    vi.mocked(getSystemStatus).mockResolvedValue({
      cuda_available: true,
      supported_upload_formats: [".tif"],
    });
    renderLibrary();

    expect(await screen.findByText("GPU: CUDA")).toBeInTheDocument();
  });

  it("shows the workflow guide on a first visit and remembers it was dismissed", async () => {
    window.localStorage.clear();
    const user = userEvent.setup();
    const { unmount } = renderLibrary();

    expect(await screen.findByText("How QuantEM works")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Got it" }));
    expect(screen.queryByText("How QuantEM works")).toBeNull();

    unmount();
    renderLibrary();
    await screen.findByRole("heading", { name: "Library" });
    expect(screen.queryByText("How QuantEM works")).toBeNull();
  });

  it("can reopen the guide from the header", async () => {
    const user = userEvent.setup();
    renderLibrary();

    await user.click(await screen.findByRole("button", { name: "How this works" }));
    expect(screen.getByText("How QuantEM works")).toBeInTheDocument();
  });

  /**
   * Owner ask #1. This button used to call `openImportPanel`, which set
   * `defaultExpanded` and remounted the panel -- so "Import an image" expanded
   * an accordion and left a bare `<input type=file>` still to be clicked.
   *
   * The button this now finds is the **empty state's**, not the header's: the
   * header no longer has one, by a later owner ruling, because the import panel
   * sits immediately below it and already takes a click or a dropped file. The
   * empty state keeps its button because there is no library on screen to point
   * at instead.
   */
  it("opens the OS file picker from Import an image, with no accordion in between", async () => {
    const user = userEvent.setup();
    renderLibrary();

    const input = await screen.findByLabelText(/image file/i);
    const clicked = vi.spyOn(input as HTMLInputElement, "click");

    await user.click(screen.getAllByRole("button", { name: "Import an image" })[0]);

    expect(clicked).toHaveBeenCalledTimes(1);
  });

  it("accepts a file dropped anywhere on the page, not only on the panel", async () => {
    renderLibrary();
    await screen.findByTestId("import-drop-zone");

    // The page container, three levels above the panel.
    const page = document.querySelector(".min-h-screen") as HTMLElement;
    fireEvent.drop(page, {
      dataTransfer: {
        files: [new File(["x"], "dropped-on-page.tif", { type: "image/tiff" })],
        types: ["Files"],
      },
    });

    expect(await screen.findByText("dropped-on-page.tif")).toBeInTheDocument();
  });

  /**
   * The confirmed bug: `LibraryPage` never sent `ordering`, so
   * `_filtered_asset_queryset` fell back to `display_name` and returned the
   * alphabetically first 60 rows. With 62 assets a new import could be absent
   * from page 1 entirely, under a sort control reading "Imported / Descending".
   */
  describe("which rows the server is asked for", () => {
    it("sends the ordering the sort control is showing", async () => {
      renderLibrary();

      await waitFor(() => expect(getHomeEntryPage).toHaveBeenCalled());
      expect(getHomeEntryPage).toHaveBeenCalledWith(
        expect.objectContaining({ ordering: "-created_at" })
      );
    });

    it("changes the ordering with the control, not only the local sort", async () => {
      const user = userEvent.setup();
      renderLibrary();
      await waitFor(() => expect(getHomeEntryPage).toHaveBeenCalled());

      await user.selectOptions(screen.getByLabelText("Sort field"), "display_name");
      await user.selectOptions(screen.getByLabelText("Sort direction"), "asc");

      await waitFor(() => {
        expect(getHomeEntryPage).toHaveBeenLastCalledWith(
          expect.objectContaining({ ordering: "display_name" })
        );
      });
    });

    /**
     * The sort control offers only sorts that exist.
     *
     * "Status" was offered and could not be honoured: `ASSET_ORDERINGS` has no
     * status key, an unknown `ordering` is *silently* replaced by
     * `display_name` server-side, so the page fetched the newest 60 rows and
     * rearranged those 60 by raw `preprocess_stage` string. On a library of
     * 212 that is "reorder page 1" under a label promising to order the
     * library, in an order (alphabetical over `CANCELLED`, `ENCODING`,
     * `FAILED`, `NONE`, `SAM`) that matches neither the cards nor anything a
     * user would ask for.
     */
    it("does not offer a sort the server cannot do", async () => {
      renderLibrary();
      await waitFor(() => expect(getHomeEntryPage).toHaveBeenCalled());

      const control = screen.getByLabelText("Sort field");
      expect(
        Array.from(control.querySelectorAll("option")).map(
          (option) => option.value
        )
      ).toEqual(["display_name", "created_at", "updated_at"]);
      expect(screen.queryByRole("option", { name: "Status" })).toBeNull();
    });

    it("ignores a Status preference stored before it was removed", async () => {
      window.localStorage.setItem(
        "quantem-library-controls-v1",
        JSON.stringify({ sortField: "status", sortDirection: "desc" })
      );
      renderLibrary();

      await waitFor(() => expect(getHomeEntryPage).toHaveBeenCalled());
      expect(getHomeEntryPage).toHaveBeenCalledWith(
        expect.objectContaining({ ordering: "-created_at" })
      );
      expect(screen.getByLabelText("Sort field")).toHaveValue("created_at");
    });
  });

  /**
   * "It seemed to have failed (reverted back to upload button) but took several
   * seconds for the image to eventually appear below that as selectable."
   */
  describe("after an import", () => {
    it("confirms the import and says where the image went", async () => {
      renderLibrary();
      const confirmation = await importOneFile();

      expect(confirmation).toHaveTextContent("Imported grid2");
      expect(confirmation).toHaveTextContent("It is the first card below");
    });

    it("puts the card on screen immediately, without waiting for a refetch", async () => {
      // The list keeps answering with the *old* library, exactly as it does
      // when the new row is on page 2 of an alphabetical ordering, or when the
      // post-upload refetch collides with the 3 s poll and is dropped.
      vi.mocked(getHomeEntryPage).mockResolvedValue(
        pageOf([makeEntry({ id: "asset-existing", display_name: "Aardvark" })])
      );
      renderLibrary();
      await screen.findByText("Aardvark");

      await importOneFile();

      const cards = screen.getAllByRole("article");
      expect(within(cards[0]).getByRole("link").textContent).toBe("grid2");
      expect(within(cards[0]).getByText("Just imported")).toBeInTheDocument();
    });

    it("keeps the import first even when the sort would put it last", async () => {
      vi.mocked(getHomeEntryPage).mockResolvedValue(
        pageOf([makeEntry({ id: "asset-existing", display_name: "Aardvark" })])
      );
      const user = userEvent.setup();
      renderLibrary();
      await screen.findByText("Aardvark");
      await importOneFile();

      await user.selectOptions(screen.getByLabelText("Sort field"), "display_name");
      await user.selectOptions(screen.getByLabelText("Sort direction"), "asc");

      await waitFor(() => {
        const cards = screen.getAllByRole("article");
        expect(within(cards[0]).getByRole("link").textContent).toBe("grid2");
      });
    });

    /**
     * The card's badge is the only place the user can see the ~100 s of
     * preparation happening, and it used to read "NGFF pending" for all of it.
     */
    it("counts the preparation up on the card and in the confirmation", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      renderLibrary();
      await importOneFile();

      vi.mocked(getAsset).mockResolvedValue(
        makeUploadedAsset({ preprocess_stage: "ENCODING", preprocess_progress: 62 })
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1100);
      });

      expect(await screen.findByText("Preparing 62%")).toBeInTheDocument();
      expect(screen.getByTestId("import-confirmation")).toHaveTextContent(
        "preparing 62%"
      );
    });

    it("announces the hand-off into the viewer before it happens", async () => {
      renderLibrary();
      const confirmation = await importOneFile();

      expect(confirmation).toHaveTextContent(
        "I will open it here when it is ready"
      );
      expect(navigateMock).not.toHaveBeenCalled();
    });

    it("counts down out loud and then opens the viewer", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      renderLibrary();
      await importOneFile();

      vi.mocked(getAsset).mockResolvedValue(
        makeUploadedAsset({ preprocess_stage: "DONE", ngff_ready: true })
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1100);
      });

      expect(screen.getByTestId("import-confirmation")).toHaveTextContent(
        "Opening it in 5…"
      );
      expect(navigateMock).not.toHaveBeenCalled();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(5200);
      });

      expect(navigateMock).toHaveBeenCalledWith("/assets/asset-new/viewer");
    });

    /**
     * The measured surprise: on a 475 MP image the route change arrived ~100 s
     * after the import, silently, from an effect the user could neither see nor
     * stop.
     */
    it("can be told to stay in the library, and then stays", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      renderLibrary();
      await importOneFile();

      await user.click(screen.getByRole("button", { name: "Stay in the library" }));

      vi.mocked(getAsset).mockResolvedValue(
        makeUploadedAsset({ preprocess_stage: "DONE", ngff_ready: true })
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(12_000);
      });

      expect(navigateMock).not.toHaveBeenCalled();
      expect(screen.getByTestId("import-confirmation")).toHaveTextContent(
        "open it when you want"
      );
    });

    it("still opens on request after the countdown was stopped", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      const user = userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
      renderLibrary();
      await importOneFile();
      await user.click(screen.getByRole("button", { name: "Stay in the library" }));

      vi.mocked(getAsset).mockResolvedValue(
        makeUploadedAsset({ preprocess_stage: "DONE", ngff_ready: true })
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1100);
      });
      await user.click(screen.getByRole("button", { name: "Open it now" }));

      expect(navigateMock).toHaveBeenCalledWith("/assets/asset-new/viewer");
    });

    it("says why an import could not be prepared instead of opening it", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      renderLibrary();
      await importOneFile();

      vi.mocked(getAsset).mockResolvedValue(
        makeUploadedAsset({
          preprocess_stage: "FAILED",
          preprocess_error: "Out of memory: Image is too large to process.",
        })
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1100);
      });

      const confirmation = screen.getByTestId("import-confirmation");
      expect(confirmation).toHaveTextContent(
        "Out of memory: Image is too large to process."
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(10_000);
      });
      expect(navigateMock).not.toHaveBeenCalled();
    });
  });

  /**
   * A plate, not a picture (plan §1.12).
   *
   * The page used to hold exactly one `justImported`, so a second import
   * replaced the first: pin, highlight, confirmation and the 1 s status poll
   * all pointed at the last file only. And the auto-open into the viewer is
   * written for "*the* image" -- with forty there is no defensible answer to
   * which one, and jumping into any of them mid-queue is precisely the
   * unannounced route change this screen spent a package removing.
   */
  describe("importing several images at once", () => {
    it("pins every image of a batch, in the order they landed", async () => {
      vi.mocked(getHomeEntryPage).mockResolvedValue(
        pageOf([makeEntry({ id: "asset-existing", display_name: "Aardvark" })])
      );
      vi.mocked(uploadAsset)
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-a", display_name: "grid-a" })
        )
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-b", display_name: "grid-b" })
        )
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-c", display_name: "grid-c" })
        );
      renderLibrary();
      await screen.findByText("Aardvark");

      await importFiles(["a.tif", "b.tif", "c.tif"]);

      await waitFor(() => expect(uploadAsset).toHaveBeenCalledTimes(3));
      const cards = screen.getAllByRole("article");
      expect(
        cards.slice(0, 3).map((card) => within(card).getByRole("link").textContent)
      ).toEqual(["grid-a", "grid-b", "grid-c"]);
      // Every one of them is badged, not just the last.
      expect(screen.getAllByText("Just imported")).toHaveLength(3);
    });

    it("stays in the library rather than opening one of the batch", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      vi.mocked(uploadAsset)
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-a", display_name: "grid-a" })
        )
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-b", display_name: "grid-b" })
        );
      // Both finish preparing, which for a single import is exactly when the
      // countdown starts and the route changes.
      vi.mocked(getAsset).mockImplementation(async (assetId: string) =>
        makeUploadedAsset({
          id: assetId,
          preprocess_stage: "DONE",
          ngff_ready: true,
        })
      );
      renderLibrary();

      const confirmation = await importFiles(["a.tif", "b.tif"]);

      expect(confirmation).toHaveTextContent("Imported 2 images");
      expect(confirmation).toHaveTextContent("They are the first cards below");
      await act(async () => {
        await vi.advanceTimersByTimeAsync(12_000);
      });
      expect(navigateMock).not.toHaveBeenCalled();
      expect(
        screen.queryByRole("button", { name: /open it/i })
      ).not.toBeInTheDocument();
    });

    /**
     * The race the batch position exists for.
     *
     * A plate is rarely uniform: a 300 KB PNG can be imported and fully
     * prepared inside two seconds while the 2 GB mosaic behind it is still
     * uploading. At that instant exactly one image has landed, so "is this a
     * lone import?" cannot be answered by counting what has arrived — and
     * getting it wrong means the app navigates into image 1 and abandons the
     * queue the user is watching. Only the panel running the queue knows, so it
     * says.
     */
    it("does not open the first image while the rest of the batch is still uploading", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      // A holder rather than a bare `let`: assigning inside a callback narrows
      // a `let` to `never` by the time the test reads it back.
      const second: { release: (() => void) | null } = { release: null };
      vi.mocked(uploadAsset)
        .mockResolvedValueOnce(
          makeUploadedAsset({
            id: "asset-a",
            display_name: "grid-a",
            preprocess_stage: "DONE",
            ngff_ready: true,
          })
        )
        .mockImplementationOnce(
          () =>
            new Promise((resolve) => {
              second.release = () =>
                resolve(makeUploadedAsset({ id: "asset-b", display_name: "grid-b" }));
            })
        );
      vi.mocked(getAsset).mockImplementation(async (assetId: string) =>
        makeUploadedAsset({ id: assetId, preprocess_stage: "DONE", ngff_ready: true })
      );
      renderLibrary();

      const zone = await screen.findByTestId("import-drop-zone");
      fireEvent.drop(zone, {
        dataTransfer: {
          files: ["a.png", "b.tif"].map(
            (name) => new File([new Uint8Array(16)], name, { type: "image/tiff" })
          ),
          types: ["Files"],
        },
      });
      await waitFor(() =>
        expect(screen.getAllByTestId("import-file-row")).toHaveLength(2)
      );
      await act(async () => {
        fireEvent.submit(screen.getByTestId("import-form"));
      });

      // Long enough for the five-second countdown to have run twice over.
      await act(async () => {
        await vi.advanceTimersByTimeAsync(12_000);
      });

      expect(navigateMock).not.toHaveBeenCalled();
      expect(screen.getByTestId("import-confirmation")).not.toHaveTextContent(
        /Opening it in/
      );
      second.release?.();
    });

    it("counts what is ready and what is still preparing", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      vi.mocked(uploadAsset)
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-a", display_name: "grid-a" })
        )
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-b", display_name: "grid-b" })
        );
      // One image finishes, the other is still encoding: the strip has to be
      // able to say both at once.
      vi.mocked(getAsset).mockImplementation(async (assetId: string) =>
        assetId === "asset-a"
          ? makeUploadedAsset({
              id: assetId,
              preprocess_stage: "DONE",
              ngff_ready: true,
            })
          : makeUploadedAsset({
              id: assetId,
              preprocess_stage: "ENCODING",
              preprocess_progress: 40,
            })
      );
      renderLibrary();
      await importFiles(["a.tif", "b.tif"]);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1200);
      });

      expect(screen.getByTestId("import-confirmation")).toHaveTextContent(
        "1 ready · 1 still preparing"
      );
    });

    it("names the ones that could not be prepared", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      vi.mocked(uploadAsset)
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-a", display_name: "grid-a" })
        )
        .mockResolvedValueOnce(
          makeUploadedAsset({ id: "asset-b", display_name: "grid-b" })
        );
      vi.mocked(getAsset).mockImplementation(async (assetId: string) =>
        assetId === "asset-b"
          ? makeUploadedAsset({
              id: assetId,
              display_name: "grid-b",
              preprocess_stage: "FAILED",
              preprocess_error: "Out of memory: Image is too large to process.",
            })
          : makeUploadedAsset({
              id: assetId,
              display_name: "grid-a",
              preprocess_stage: "DONE",
              ngff_ready: true,
            })
      );
      renderLibrary();
      await importFiles(["a.tif", "b.tif"]);

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1200);
      });

      const confirmation = screen.getByTestId("import-confirmation");
      expect(confirmation).toHaveTextContent(
        "grid-b could not be prepared: Out of memory: Image is too large to process."
      );
      expect(confirmation).toHaveTextContent("1 failed");
    });
  });

  /**
   * `refetchEntries` -> `loadEntryPage(0, "replace")` -> `setEntriesLoading(true)`,
   * and `entriesLoading` gated the whole grid. So every 3 s poll while anything
   * was preprocessing unmounted every card, blanked the page to "Loading
   * images…" and remounted them, re-requesting every thumbnail -- for the
   * entire ~100 s of a large import, and again at the exact moment the user was
   * looking for their new card.
   */
  describe("polling while an image is being prepared", () => {
    it("does not blank the grid on a background refetch", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      const processing = makeEntry({
        id: "asset-processing",
        display_name: "Still cooking",
        preprocess_stage: "ENCODING",
        preprocess_progress: 40,
        ngff_ready: false,
      });
      vi.mocked(getHomeEntryPage).mockResolvedValue(pageOf([processing]));
      renderLibrary();

      const card = await screen.findByText("Still cooking");
      const callsBefore = vi.mocked(getHomeEntryPage).mock.calls.length;

      // The poll's response is held open, so the in-flight state is really
      // rendered. With an instantly-resolved mock React can coalesce the
      // loading flag's rise and fall into one commit and the flash -- which is
      // what the user sees for a real ~60 ms request -- never appears.
      let resolvePoll: (page: HomeEntryPage) => void = () => {};
      vi.mocked(getHomeEntryPage).mockReturnValue(
        new Promise<HomeEntryPage>((resolve) => {
          resolvePoll = resolve;
        })
      );

      await act(async () => {
        await vi.advanceTimersByTimeAsync(3200);
      });

      expect(vi.mocked(getHomeEntryPage).mock.calls.length).toBeGreaterThan(
        callsBefore
      );
      // Mid-refetch: the grid is still here and says so quietly.
      expect(screen.queryByText("Loading images...")).not.toBeInTheDocument();
      expect(screen.getByText("Still cooking")).toBe(card);
      expect(screen.getByText(/updating…/)).toBeInTheDocument();

      await act(async () => {
        resolvePoll(pageOf([processing]));
      });

      // Still the same DOM node afterwards: a remounted card re-creates its
      // `<img>` and re-requests the thumbnail, which is what made the page
      // flash every three seconds for the whole of a 100 s import.
      expect(screen.getByText("Still cooking")).toBe(card);
      await waitFor(() =>
        expect(screen.queryByText(/updating…/)).not.toBeInTheDocument()
      );
    });

    it("still shows a first-load state when there is nothing to keep", async () => {
      let resolvePage: (page: HomeEntryPage) => void = () => {};
      vi.mocked(getHomeEntryPage).mockReturnValue(
        new Promise<HomeEntryPage>((resolve) => {
          resolvePage = resolve;
        })
      );
      renderLibrary();

      expect(await screen.findByText("Loading images...")).toBeInTheDocument();
      await act(async () => {
        resolvePage(EMPTY_PAGE);
      });
      await waitFor(() =>
        expect(screen.queryByText("Loading images...")).not.toBeInTheDocument()
      );
    });
  });

  /**
   * The grouping layer, from the library's side.
   *
   * The governing rule is that an unorganised library is a legitimate steady
   * state. Nothing here may appear, nag, or block until the user has made an
   * experiment, and the default msw handler returns none -- which is why every
   * other test in this file still describes the screen correctly.
   */
  describe("experiments and datasets", () => {
    function withExperiments() {
      server.use(
        http.get("http://127.0.0.1:8000/api/experiments/", () =>
          HttpResponse.json([
            {
              id: "exp-1",
              name: "Fasted cohort",
              notes: "",
              datasets: [
                {
                  id: "set-1",
                  experiment: "exp-1",
                  name: "Liver 24h",
                  notes: "",
                  asset_count: 1,
                  created_at: null,
                  updated_at: null,
                },
              ],
              asset_count: 1,
              ungrouped_asset_count: 0,
              created_at: null,
              updated_at: null,
            },
          ])
        )
      );
    }

    it("shows no filter at all until something has been organised", async () => {
      renderLibrary();
      await waitFor(() => expect(getHomeEntryPage).toHaveBeenCalled());

      expect(screen.queryByLabelText("Experiment")).not.toBeInTheDocument();
      expect(screen.queryByLabelText("Dataset")).not.toBeInTheDocument();
      // And no prompt to go and make one. Unorganised is not unfinished.
      expect(screen.queryByText(/organise/i)).not.toBeInTheDocument();
    });

    /**
     * Narrowing has to happen server-side. The page holds one window of sixty
     * rows, and filtering a window is not filtering a library -- the same
     * defect the Status sort had.
     */
    it("asks the server for the experiment, not the page", async () => {
      withExperiments();
      const user = userEvent.setup();
      renderLibrary();
      await screen.findByLabelText("Experiment");

      await user.selectOptions(screen.getByLabelText("Experiment"), "exp-1");

      await waitFor(() =>
        expect(getHomeEntryPage).toHaveBeenLastCalledWith(
          expect.objectContaining({ experiment: "exp-1" })
        )
      );
    });

    it("can ask for the images that are in no experiment", async () => {
      withExperiments();
      const user = userEvent.setup();
      renderLibrary();
      await screen.findByLabelText("Experiment");

      await user.selectOptions(
        screen.getByLabelText("Experiment"),
        "none"
      );

      await waitFor(() =>
        expect(getHomeEntryPage).toHaveBeenLastCalledWith(
          expect.objectContaining({ experiment: "none" })
        )
      );
    });

    it("offers no tick boxes until the user asks to select", async () => {
      vi.mocked(getHomeEntryPage).mockResolvedValue(
        pageOf([makeEntry({ id: "asset-1", display_name: "Liver 01" })])
      );
      const user = userEvent.setup();
      renderLibrary();
      await screen.findByText("Liver 01");

      expect(screen.queryByLabelText("Select Liver 01")).not.toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "Select images" }));

      expect(await screen.findByLabelText("Select Liver 01")).toBeInTheDocument();
    });

    it("shows the assignment bar once an image is ticked", async () => {
      vi.mocked(getHomeEntryPage).mockResolvedValue(
        pageOf([makeEntry({ id: "asset-1", display_name: "Liver 01" })])
      );
      const user = userEvent.setup();
      renderLibrary();
      await screen.findByText("Liver 01");

      await user.click(screen.getByRole("button", { name: "Select images" }));
      expect(
        screen.queryByTestId("library-selection-bar")
      ).not.toBeInTheDocument();

      await user.click(screen.getByLabelText("Select Liver 01"));

      expect(
        await screen.findByTestId("library-selection-bar")
      ).toBeInTheDocument();
      expect(screen.getByText("1 image selected")).toBeInTheDocument();
    });

    it("groups the loaded images by dataset when asked", async () => {
      withExperiments();
      vi.mocked(getHomeEntryPage).mockResolvedValue(
        pageOf([
          makeEntry({
            id: "asset-1",
            display_name: "Liver 01",
            experiment_id: "exp-1",
            experiment_name: "Fasted cohort",
            dataset_ids: ["set-1"],
            dataset_names: ["Liver 24h"],
          }),
          makeEntry({ id: "asset-2", display_name: "Loose 01" }),
        ])
      );
      const user = userEvent.setup();
      renderLibrary();
      await screen.findByLabelText("Experiment");

      await user.click(screen.getByLabelText("Group by dataset"));

      expect(
        await screen.findByRole("heading", { name: /Liver 24h/ })
      ).toBeInTheDocument();
      // The unassigned images keep a section of their own rather than
      // disappearing out of the grouped view.
      expect(
        screen.getByRole("heading", { name: /Not in an experiment/ })
      ).toBeInTheDocument();
    });

    /**
     * "Delete" over a group of images reads as "delete the images". It is not,
     * and the dialog has to say so before the click, with the count in it.
     */
    it("promises that deleting an experiment keeps the images", async () => {
      withExperiments();
      const user = userEvent.setup();
      renderLibrary();
      await screen.findByLabelText("Experiment");

      await user.selectOptions(screen.getByLabelText("Experiment"), "exp-1");
      await user.click(
        await screen.findByRole("button", { name: "Delete experiment" })
      );

      expect(
        await screen.findByText(
          /Its 1 image stays in the library and becomes unassigned\..*No image files are deleted\./
        )
      ).toBeInTheDocument();
    });
  });
});
