fn main() {
    tauri_build::try_build(
        tauri_build::Attributes::new().app_manifest(
            tauri_build::AppManifest::new()
                .commands(&["save_text_file", "save_url_file"]),
        ),
    )
    .expect("failed to prepare the QuantEM desktop build")
}
