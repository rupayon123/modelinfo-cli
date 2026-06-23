from rich.console import Console

from modelinfo import ui


def _recording_console(monkeypatch) -> Console:
    console = Console(record=True, color_system=None, width=120)
    monkeypatch.setattr(ui, "console", console)
    return console


def _footprint(total_memory_bytes: float = 1024**3):
    return {
        "total_params": 20,
        "primary_dtype": "F16",
        "base_memory_bytes": 512 * 1024**2,
        "kv_cache_bytes": 128 * 1024**2,
        "overhead_bytes": 64 * 1024**2,
        "total_memory_bytes": total_memory_bytes,
        "kv_is_estimate": False,
    }


def test_print_model_info_renders_summary_and_tensor_groups(monkeypatch):
    console = _recording_console(monkeypatch)
    tensors = {
        "model.layers.0.mlp.down_proj.weight": {"shape": [2, 3], "dtype": "F16"},
        "model.layers.1.mlp.down_proj.weight": {"shape": [2, 3], "dtype": "F16"},
        "lm_head.weight": {"shape": [4, 5], "dtype": "F32"},
    }

    ui.print_model_info(
        "safetensors",
        "MockArch",
        len(tensors),
        _footprint(),
        disk_size=2 * 1024**3,
        context_length=2048,
        is_default_context=True,
        tensors=tensors,
        max_context=4096,
        max_vram_gb=8.0,
        gpu_name="Test GPU",
    )

    output = console.export_text()

    assert "Format:" in output
    assert "safetensors" in output
    assert "Architecture:" in output
    assert "MockArch" in output
    assert "VRAM (est):" in output
    assert "Weights:" in output
    assert "KV Cache:" in output
    assert "Top Tensors by Size:" in output
    assert "2x model.layers.[N].mlp.down_proj.weight" in output
    assert "lm_head.weight" in output


def test_print_model_info_warns_about_missing_shards(monkeypatch):
    console = _recording_console(monkeypatch)
    tensors = {"__metadata__": {"missing_shards": 2, "total_shards": 5}}

    ui.print_model_info(
        "safetensors",
        "MockArch",
        0,
        _footprint(total_memory_bytes=0),
        disk_size=0,
        context_length=0,
        is_default_context=True,
        tensors=tensors,
    )

    output = console.export_text()

    assert "UNKNOWN (Missing Shards)" in output
    assert "WARNING: Partial Model. Missing 2 of 5 shards on disk." in output


def test_print_compare_info_renders_fit_column(monkeypatch):
    console = _recording_console(monkeypatch)
    models = [
        (
            "tiny-model",
            {
                "context_length": 2048,
                "footprint": {
                    "total_params": 20,
                    "primary_dtype": "F16",
                    "total_memory_bytes": 512 * 1024**2,
                },
            },
        )
    ]

    ui.print_compare_info(models, max_vram_gb=8.0, gpu_name="Test GPU")

    output = console.export_text()

    assert "Model" in output
    assert "tiny-model" in output
    assert "Fits" in output
