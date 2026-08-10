/**
 * Configuration constants for the frontend application.
 */

/**
 * Douglas-Peucker tolerance for **drawing on screen**, and nothing else.
 *
 * `withSmoothedSegmentGeometry` uses it to build `smoothed_geometry_coords`,
 * a display copy of an outline; `geometry_coords` -- the outline the server
 * stored and measured -- is never replaced by it.
 *
 * There is deliberately **no counterpart on the submit side**. A `SIMPLIFY_POLYGONS`
 * flag used to run the same simplifier at 1.0 px over every ring on its way to
 * `confirm-batch` and `remove-area`, so the outline the server measured was not
 * the outline the user drew. Measured through `quantem.seg_core.rasterize` --
 * the server's own pixel-centre fill -- against the ring as drawn:
 *
 * | shape (freehand, ~0.75 px pointer samples) | 0.25 px | 0.5 px | 1.0 px |
 * |--------------------------------------------|---------|--------|--------|
 * | 8 px object, sampled every 2 px            |  0.0%   | -6.7%  | -15.6% |
 * | 20 px lipid droplet                        |  0.0%   |  0.0%  |  -7.8% |
 * | 60 px mitochondrion                        | -1.0%   | -2.1%  |  -2.5% |
 * | 200 px object                              | -0.1%   | -0.5%  |  -0.6% |
 * | 20 px square (the shape the old case used) |         |        |  -9.8% |
 *
 * and against a brush ring, which is the exact pixel boundary of what was
 * painted, 1.0 px went the *other* way: +7.7% on an 8 px object and +4.4% on a
 * 20 px droplet. So it is not a bias that could be corrected for downstream --
 * two outlines of the same organelle drawn with different tools move in
 * opposite directions, and by more on the small objects this app is mostly
 * used on. It undid the whole point of the pixel-centre rasteriser, which
 * exists so a hand-drawn object and a model-found object of the same shape
 * measure the same.
 *
 * Nothing was bought with it that mattered: the display path above already
 * simplifies for rendering, and the largest payload it saved is small -- a
 * tissue mask brushed over a 2000 px circle posts 4893 points / 52 KB raw.
 * Anything sent to be measured goes as drawn.
 */
export const SEGMENT_SMOOTHING_TOLERANCE = 1.0;
export const SEGMENT_SMOOTHING_VIEWPORT_DIMENSION_THRESHOLD_PX = 4000;
export const SEGMENT_VIEWPORT_FETCH_DEBOUNCE_MS = 120;

export type AppRuntimeConfig = {
  apiBaseUrl?: string;
  dev?: boolean;
};

type AppConfigWindow = Window & {
  __APP_CONFIG__?: AppRuntimeConfig;
};

export function getRuntimeConfig(): AppRuntimeConfig {
  const win = window as AppConfigWindow;
  return win.__APP_CONFIG__ || {};
}
