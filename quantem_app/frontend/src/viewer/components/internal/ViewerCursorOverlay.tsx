export function ViewerCursorOverlay(config: {
  hoverBadge?: { point: import("@/utils/geometry").Point | null; count: number };
  hoverBadgeStyle: React.CSSProperties;
  overlayCursorState: {
    x: number;
    y: number;
    outerSize: number;
    borderWidth: number;
    visible: boolean;
    variant: "brush" | "target";
  };
  brushColor: string;
}) {
  const { hoverBadge, hoverBadgeStyle, overlayCursorState, brushColor } = config;

  return (
    <>
      {hoverBadge && hoverBadge.count > 1 ? (
        <div className="hover-count-badge" style={hoverBadgeStyle}>
          {hoverBadge.count}
        </div>
      ) : null}
      {overlayCursorState.visible && overlayCursorState.variant === "brush" ? (
        <div
          className="brush-cursor"
          style={{
            position: "absolute",
            left: `${overlayCursorState.x}px`,
            top: `${overlayCursorState.y}px`,
            width: `${overlayCursorState.outerSize}px`,
            height: `${overlayCursorState.outerSize}px`,
            transform: "translate(-50%, -50%)",
            borderRadius: "50%",
            border: `${overlayCursorState.borderWidth}px solid ${brushColor}`,
            backgroundColor: "transparent",
            pointerEvents: "none",
            zIndex: 1000,
            boxSizing: "border-box",
          }}
        />
      ) : null}
      {overlayCursorState.visible && overlayCursorState.variant === "target" ? (
        <div
          data-testid="target-cursor"
          style={{
            position: "absolute",
            left: `${overlayCursorState.x}px`,
            top: `${overlayCursorState.y}px`,
            width: `${overlayCursorState.outerSize}px`,
            height: `${overlayCursorState.outerSize}px`,
            transform: "translate(-50%, -50%)",
            pointerEvents: "none",
            zIndex: 1000,
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              borderRadius: "50%",
              border: `${overlayCursorState.borderWidth}px solid #facc15`,
              boxSizing: "border-box",
            }}
          />
          <div
            style={{
              position: "absolute",
              left: "50%",
              top: 0,
              bottom: 0,
              width: "1px",
              transform: "translateX(-50%)",
              backgroundColor: "#facc15",
              opacity: 0.95,
            }}
          />
          <div
            style={{
              position: "absolute",
              top: "50%",
              left: 0,
              right: 0,
              height: "1px",
              transform: "translateY(-50%)",
              backgroundColor: "#facc15",
              opacity: 0.95,
            }}
          />
        </div>
      ) : null}
    </>
  );
}

