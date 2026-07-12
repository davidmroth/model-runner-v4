#!/usr/bin/env python3
"""Patch test_dflash path to support native mmproj vision.

Run on ai.local as david:
  python3 /tmp/patch-vision-test-dflash.py
"""
import pathlib, shutil, sys

SRC = pathlib.Path("/media/data/projects/lucebox-hub-src/server")

def patch(path: pathlib.Path, replacements: list[tuple[str, str]], label: str):
    code = path.read_text()
    bak = pathlib.Path(str(path) + ".bak_vision")
    if not bak.exists():
        shutil.copy(path, bak)
    for old, new in replacements:
        assert old in code, f"{label}: anchor not found:\n{old!r}"
        code = code.replace(old, new, 1)
    path.write_text(code)
    print(f"OK: patched {path.name}")

# ── 1. layer_split_daemon_loop.h ─────────────────────────────────────────────
patch(
    SRC / "src/qwen35/layer_split_daemon_loop.h",
    [(
        "    int draft_ctx_max    = 2048;\n"
        "};",
        "    int draft_ctx_max    = 2048;\n"
        "\n"
        "    // Native mmproj vision — must match DFLASH_MMPROJ env var\n"
        "    const char * mmproj_path    = nullptr;  // nullptr = vision disabled\n"
        "    bool         mmproj_use_gpu = true;     // DFLASH_MMPROJ_NO_OFFLOAD=0 sets false\n"
        "};"
    )],
    "layer_split_daemon_loop.h"
)

# ── 2. layer_split_daemon_loop.cpp ───────────────────────────────────────────
patch(
    SRC / "src/qwen35/layer_split_daemon_loop.cpp",
    [(
        "    args.fa_window  = cfg.fa_window;\n"
        "    args.kq_stride_pad = cfg.kq_stride_pad;",
        "    args.fa_window  = cfg.fa_window;\n"
        "    args.kq_stride_pad = cfg.kq_stride_pad;\n"
        "    args.mmproj_path    = cfg.mmproj_path;\n"
        "    args.mmproj_use_gpu = cfg.mmproj_use_gpu;"
    )],
    "layer_split_daemon_loop.cpp"
)

# ── 3. test_dflash.cpp ────────────────────────────────────────────────────────
# 3a. Add global mmproj variables (near other globals, after g_fa_window)
patch(
    SRC / "test/test_dflash.cpp",
    [(
        "static int run_target_layer_split_daemon(\n"
        "        const char * target_path,\n"
        "        const char * draft_path,\n"
        "        const std::vector<int> & target_gpus,\n"
        "        const std::vector<double> & split_weights,\n"
        "        int draft_gpu,\n"
        "        bool load_draft,\n"
        "        bool run_dflash,\n"
        "        int max_ctx,\n"
        "        int max_verify_tokens,\n"
        "        bool peer_access,\n"
        "        int stream_fd) {",
        # New signature adds mmproj_path + mmproj_use_gpu
        "static int run_target_layer_split_daemon(\n"
        "        const char * target_path,\n"
        "        const char * draft_path,\n"
        "        const std::vector<int> & target_gpus,\n"
        "        const std::vector<double> & split_weights,\n"
        "        int draft_gpu,\n"
        "        bool load_draft,\n"
        "        bool run_dflash,\n"
        "        int max_ctx,\n"
        "        int max_verify_tokens,\n"
        "        bool peer_access,\n"
        "        int stream_fd,\n"
        "        const char * mmproj_path = nullptr,\n"
        "        bool mmproj_use_gpu = true) {"
    )],
    "test_dflash.cpp (signature)"
)

# 3b. Set mmproj config in the body of run_target_layer_split_daemon
patch(
    SRC / "test/test_dflash.cpp",
    [(
        "    cfg.kq_stride_pad = g_kq_stride_pad;\n"
        "    cfg.fa_window = g_fa_window;\n"
        "    cfg.draft_ctx_max = g_draft_ctx_max;\n"
        "    return run_layer_split_daemon(cfg);",
        "    cfg.kq_stride_pad = g_kq_stride_pad;\n"
        "    cfg.fa_window = g_fa_window;\n"
        "    cfg.draft_ctx_max = g_draft_ctx_max;\n"
        "    cfg.mmproj_path    = mmproj_path;\n"
        "    cfg.mmproj_use_gpu = mmproj_use_gpu;\n"
        "    return run_layer_split_daemon(cfg);"
    )],
    "test_dflash.cpp (body)"
)

# 3c. Wire mmproj at the call site
patch(
    SRC / "test/test_dflash.cpp",
    [(
        "            return run_target_layer_split_daemon(\n"
        "                lsargs.target_path, lsargs.draft_path,\n"
        "                lsargs.device.layer_split_gpus,\n"
        "                lsargs.device.layer_split_weights,\n"
        "                lsargs.draft_gpu,\n"
        "                lsargs.load_draft,\n"
        "                lsargs.run_dflash,\n"
        "                lsargs.device.max_ctx,\n"
        "                lsargs.max_verify_tokens,\n"
        "                lsargs.device.peer_access,\n"
        "                lsargs.stream_fd);",
        "            return run_target_layer_split_daemon(\n"
        "                lsargs.target_path, lsargs.draft_path,\n"
        "                lsargs.device.layer_split_gpus,\n"
        "                lsargs.device.layer_split_weights,\n"
        "                lsargs.draft_gpu,\n"
        "                lsargs.load_draft,\n"
        "                lsargs.run_dflash,\n"
        "                lsargs.device.max_ctx,\n"
        "                lsargs.max_verify_tokens,\n"
        "                lsargs.device.peer_access,\n"
        "                lsargs.stream_fd,\n"
        "                std::getenv(\"DFLASH_MMPROJ\"),\n"
        "                std::getenv(\"DFLASH_MMPROJ_NO_OFFLOAD\") == nullptr);"
    )],
    "test_dflash.cpp (call site)"
)

# ── 4. daemon_loop.cpp — add GENERATE_MULTIMODAL command ────────────────────
# Insert the new command just before the bare-prompt handler
patch(
    SRC / "src/common/daemon_loop.cpp",
    [
        # First add vision_types.h include at top
        (
            '#include "daemon_loop.h"\n'
            '\n'
            '#include "sampler.h"',
            '#include "daemon_loop.h"\n'
            '\n'
            '#include "sampler.h"\n'
            '#include "common/vision_types.h"  // MultimodalPrompt, DecodedImage'
        ),
        # Add GENERATE_MULTIMODAL before the bare-prompt block
        (
            "        // ── Bare prompt: \"<path> <n_gen> [snap=L:S]\" ─────────────────",
            r"""        // ── GENERATE_MULTIMODAL <image_file> <text_file> <n_gen> ────────
        // image_file: raw image bytes (JPEG/PNG); text_file: marked text with
        // mtmd marker(s) at each image position; handled by prefill_multimodal.
        if (cmd == "GENERATE_MULTIMODAL") {
            std::string img_path, text_path;
            int n_gen = 0;
            iss >> img_path >> text_path >> n_gen;
            if (img_path.empty() || text_path.empty() || n_gen <= 0) {
                std::fprintf(stderr, "[daemon] GENERATE_MULTIMODAL bad args: %s\n",
                             line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            if (!backend.supports_multimodal()) {
                std::fprintf(stderr, "[daemon] GENERATE_MULTIMODAL: vision not configured "
                             "(mmproj not loaded)\n");
                std::printf("err vision_not_configured\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            // Read image bytes
            FILE * imgf = std::fopen(img_path.c_str(), "rb");
            if (!imgf) {
                std::fprintf(stderr, "[daemon] GENERATE_MULTIMODAL: cannot open image %s\n",
                             img_path.c_str());
                std::printf("err image_not_found\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            std::fseek(imgf, 0, SEEK_END);
            const long img_size = std::ftell(imgf);
            std::rewind(imgf);
            DecodedImage img;
            img.bytes.resize((size_t)img_size);
            if (std::fread(img.bytes.data(), 1, (size_t)img_size, imgf) !=
                    (size_t)img_size) {
                std::fclose(imgf);
                std::printf("err image_read_failed\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            std::fclose(imgf);
            // Read marked text
            FILE * tf = std::fopen(text_path.c_str(), "r");
            if (!tf) {
                std::fprintf(stderr, "[daemon] GENERATE_MULTIMODAL: cannot open text %s\n",
                             text_path.c_str());
                std::printf("err text_not_found\n"); std::fflush(stdout);
                io.emit(-1);
                continue;
            }
            std::fseek(tf, 0, SEEK_END);
            const long text_size = std::ftell(tf);
            std::rewind(tf);
            std::string marked_text((size_t)text_size, '\0');
            std::fread(marked_text.data(), 1, (size_t)text_size, tf);
            std::fclose(tf);
            // Build request
            MultimodalPrompt mm;
            mm.marked_text = std::move(marked_text);
            mm.images.push_back(std::move(img));
            GenerateRequest req;
            req.n_gen      = n_gen;
            req.sampler    = sampler;
            req.do_sample  = do_sample;
            req.stream     = true;
            req.multimodal = std::make_unique<MultimodalPrompt>(std::move(mm));
            auto result = backend.generate(req, io);
            if (!result.ok) {
                io.emit(-1);
                std::printf("err %s\n", result.error.c_str()); std::fflush(stdout);
                continue;
            }
            std::printf("ok N=%zu gen=%zu prefill_s=%.3f decode_s=%.3f "
                        "decode_tok_s=%.1f stream_fd=%d\n",
                        result.tokens.size(), result.tokens.size(),
                        result.prefill_s, result.decode_s,
                        result.tokens.size() / std::max(1e-9, result.decode_s),
                        io.stream_fd);
            std::fflush(stdout);
            continue;
        }

        // ── Bare prompt: "<path> <n_gen> [snap=L:S]" ─────────────────"""
        ),
    ],
    "daemon_loop.cpp"
)

print("\nAll patches applied successfully.")
print("Next: rebuild test_dflash inside the Docker builder container.")
