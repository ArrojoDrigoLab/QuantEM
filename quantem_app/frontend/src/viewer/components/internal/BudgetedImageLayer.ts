import { CompositeLayer, type DefaultProps, type Layer, type LayerContext } from "@deck.gl/core";
import { MultiscaleImageLayer } from "@hms-dbmi/viv";
import type { VivMultiscaleLayerProps } from "./vivUtils";
import { accountImageTile, overlayTileBudget } from "./overlayTileBudget";

/** Base images and auxiliary rasters share the overlay retention allowance. */
export class BudgetedImageLayer extends CompositeLayer<VivMultiscaleLayerProps> {
  static layerName = "BudgetedImageLayer";
  // Preserve Viv's shader extensions and prop comparisons. CompositeLayer's
  // empty extensions otherwise override Viv's color rendering defaults.
  static defaultProps = (MultiscaleImageLayer as unknown as {
    defaultProps: DefaultProps<VivMultiscaleLayerProps>;
  }).defaultProps;
  declare state: { resourceOwner?: symbol };

  constructor(props: VivMultiscaleLayerProps) {
    // Viv's published event types predate Deck 9's precise callback types.
    // This wrapper forwards those props unchanged to Viv.
    super(props as ConstructorParameters<typeof CompositeLayer<VivMultiscaleLayerProps>>[0]);
  }

  initializeState() {
    this.setState({ resourceOwner: Symbol("image-generation") });
  }

  updateState() {
    if (this.state.resourceOwner) {
      overlayTileBudget.register(this.state.resourceOwner, () => this.setNeedsUpdate());
    }
  }

  finalizeState(context: LayerContext) {
    if (this.state.resourceOwner) overlayTileBudget.release(this.state.resourceOwner);
    super.finalizeState(context);
  }

  renderLayers(): Layer {
    return new MultiscaleImageLayer({
      ...this.props,
      ...this.getSubLayerProps({ id: "tiles" }),
      onTileLoad: accountImageTile,
      maxCacheByteSize: overlayTileBudget.bytesPerLayer,
      maxCacheSize: 128,
    }) as Layer;
  }
}
