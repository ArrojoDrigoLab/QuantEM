#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: build_macos_dmg.sh APP_BUNDLE OUTPUT_DMG [VOLUME_NAME]" >&2
  exit 2
fi

app_bundle="$1"
output_dmg="$2"
volume_name="${3:-QuantEM}"

if [[ ! -d "$app_bundle" ]]; then
  echo "app bundle not found: $app_bundle" >&2
  exit 2
fi

temp_root="${RUNNER_TEMP:-${TMPDIR:-/tmp}}"
staging="$(mktemp -d "$temp_root/quantem-dmg.XXXXXX")"
cleanup() {
  rm -rf "$staging"
}
trap cleanup EXIT

ditto "$app_bundle" "$staging/$(basename "$app_bundle")"
ln -s /Applications "$staging/Applications"
mkdir -p "$(dirname "$output_dmg")"

for attempt in 1 2 3; do
  rm -f "$output_dmg"
  if hdiutil create -volname "$volume_name" -srcfolder "$staging" -ov -format UDZO "$output_dmg"; then
    hdiutil verify "$output_dmg"
    exit 0
  fi
  if [[ "$attempt" == 3 ]]; then
    echo "hdiutil failed to create the DMG after 3 attempts" >&2
    exit 1
  fi
  delay=$((attempt * 15))
  echo "hdiutil attempt $attempt failed; retrying in $delay seconds" >&2
  sleep "$delay"
done
