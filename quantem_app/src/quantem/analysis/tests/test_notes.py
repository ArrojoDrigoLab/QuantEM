from quantem.analysis.notes import analysis_notes, note, settings_notes


def test_structured_notes_preserve_unrecognized_codes_and_arbitrary_wording():
    message = "A future measurement has a scientific limitation."
    result = {"notes": [note("future_qualification", message)], "caveats": ["Outdated text"]}
    assert analysis_notes(result) == [message]
    result["notes"][0]["message"] = (
        "The wording changed; the qualification still belongs in the report."
    )
    assert analysis_notes(result) == [result["notes"][0]["message"]]


def test_legacy_notes_still_render_without_rewriting_saved_results():
    result = {"caveats": ["Distance-to-mito was requested but skipped: point set is empty."]}
    assert analysis_notes(result) == [
        "Distance to mito was not calculated: no points were supplied."
    ]
    assert result["caveats"][0].startswith("Distance-to-mito")


def test_settings_notes_identify_mixed_settings_without_parsing_sentences():
    notes = settings_notes(
        [
            {
                "compartment": "mito",
                "runs": [
                    {"run_id": "a", "pack_id": "quantem:mito", "threshold": 0.3},
                    {"run_id": "b", "pack_id": "quantem:mito", "threshold": 0.6},
                ],
            }
        ]
    )
    assert {item["code"] for item in notes} == {"multiple_inference_runs", "mixed_model_settings"}
    assert notes[0]["details"] == {"compartment": "mito", "count": 2}
