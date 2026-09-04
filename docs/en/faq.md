# Frequently Asked Questions (FAQ)

## Installation & Runtime Issues

<details>
<summary>Q: Startup error: "Only part of a ReadProcessMemory or WriteProcessMemory request was completed"</summary>

Try restarting the application.
</details>

<details>
<summary>Q: Startup error: "ImportError: No module named expat; use SimpleXMLTreeBuilder instead"</summary>

`Conda` may automatically update system libraries (such as `expat`) to newer versions that are incompatible with precompiled Python modules. You can resolve this by installing an earlier version of `expat`:
```bash
conda install expat=2.5.0 -y
```
</details>

<details>
<summary>Q: Slow mouse movement or delayed response in Fedora KDE / Wayland</summary>

Under the Fedora KDE desktop environment (Wayland and X11), you may experience lagging cursor movements and sluggish UI response on the canvas.

- **Solution 1**: Pass the command-line argument to force the XCB platform:
```bash
xanylabeling --qt-platform xcb
```

- **Solution 2**: Set the environment variable:
```bash
export QT_QPA_PLATFORM=xcb
xanylabeling
```

See [#1145](https://github.com/CVHub520/X-AnyLabeling/issues/1145) for details.
</details>

<details>
<summary>Q: Startup error: "Failed to execute script 'app' due to unhandled exception: 'str' object does not support item assignment"</summary>

Delete the configuration file under your user home directory (`~/.xanylabelingrc`), then restart the application.
See [#996](https://github.com/CVHub520/X-AnyLabeling/issues/996) for details.
</details>

<details>
<summary>Q: Startup error: "OPENSSL_Uplink(00007FFE6B47AC88,08): not OPENSSL_Applink"</summary>

See [#941](https://github.com/CVHub520/X-AnyLabeling/issues/941).
</details>

<details>
<summary>Q: Application crashes during runtime with `Qt5Core.dll` dependency errors</summary>

See [#907](https://github.com/CVHub520/X-AnyLabeling/issues/907).
</details>

<details>
<summary>Q: Startup warning: "Gtk-WARNING **: 17:40:30.674: Could not load a pixbuf from icon theme"</summary>

See [#893](https://github.com/CVHub520/X-AnyLabeling/issues/893).
</details>

<details>
<summary>Q: Startup error: "AttributeError: 'NoneType' object has no attribute 'items'"</summary>

Delete the `.xanylabelingrc` file in your home directory and restart the application. See [#877](https://github.com/CVHub520/X-AnyLabeling/issues/877).
</details>

<details>
<summary>Q: Startup error: "Could not locate cublasLt64_12.dll. Please make sure it is in your library path!"</summary>

- **Solution 1**: Incompatible ONNX Runtime and CUDA versions. See [#844](https://github.com/CVHub520/X-AnyLabeling/issues/844).
- **Solution 2**: If CUDA and cuDNN are not installed system-wide, see [#1014](https://github.com/CVHub520/X-AnyLabeling/issues/1014).
</details>

<details>
<summary>Q: Startup error: 'qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found'</summary>

See [#541](https://github.com/CVHub520/X-AnyLabeling/issues/541) and [#496](https://github.com/CVHub520/X-AnyLabeling/issues/496).
</details>

<details>
<summary>Q: Startup error: 'qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "/usr/lib/qt/plugins/platforms/"'</summary>

See [#761](https://github.com/CVHub520/X-AnyLabeling/issues/761).
</details>

<details>
<summary>Q: GPU version crashes immediately on launch</summary>

See [#500](https://github.com/CVHub520/X-AnyLabeling/issues/500).
</details>

<details>
<summary>Q: QStandardPaths: wrong permissions on runtime directory /run/user/1000/, 0755 instead of 0700</summary>

Add `chmod 0700 /run/user/1000/` to your `~/.bashrc` file, reload your shell, and restart the application.
</details>

<details>
<summary>Q: Startup error: "AttributeError: module 'importlib.resources' has no attribute 'files'"</summary>

Upgrade your Python version to 3.11 or higher (Python 3.12 is recommended).
</details>

---

## UI & Interaction Issues

<details>
<summary>Q: How to quickly annotate small target objects?</summary>

See [#1000](https://github.com/CVHub520/X-AnyLabeling/issues/1000).
</details>

<details>
<summary>Q: How to annotate hollow objects (polygons with holes) for segmentation tasks?</summary>

See [#991](https://github.com/CVHub520/X-AnyLabeling/issues/991).
</details>

<details>
<summary>Q: How to quickly annotate multi-object keypoint labels and grouping?</summary>

See [#982](https://github.com/CVHub520/X-AnyLabeling/issues/982).
</details>

<details>
<summary>Q: How to enable continuous drawing mode when drawing bounding boxes?</summary>

Open the `.xanylabelingrc` configuration file in your home directory and set `auto_highlight_shape` and `auto_switch_to_edit_mode` to `False`.
See [#887](https://github.com/CVHub520/X-AnyLabeling/issues/887).
</details>

<details>
<summary>Q: The UI appears blurry on high-DPI displays</summary>

See [#811](https://github.com/CVHub520/X-AnyLabeling/issues/811).
</details>

<details>
<summary>Q: After completing an annotation, the label dialog does not pop up to assign a label name</summary>

Uncheck `Auto Use Last Label`. See [#805](https://github.com/CVHub520/X-AnyLabeling/issues/805).
</details>

<details>
<summary>Q: Overlapping objects when copying and pasting (Duplicate Polygons vs Copy & Paste Objects)</summary>

X-AnyLabeling provides two distinct copy mechanisms:

- `Ctrl+D`: Quickly duplicates objects within the current image. The new object is offset by 2 pixels by default to prevent exact overlapping.
- `Ctrl+C` and `Ctrl+V`: Intended primarily for copying objects across different images. Pasting preserves the original coordinates, so pasting within the same image will directly overlap the original object.

The system clipboard integration is disabled by default (copied objects are kept internally and can be pasted across images). To write annotations as JSON directly into the OS system clipboard, enable `Edit > Use System Clipboard`.
</details>

<details>
<summary>Q: Cannot open *.jpg, *.png, or other image files</summary>

See [#823](https://github.com/CVHub520/X-AnyLabeling/issues/823).
</details>

<details>
<summary>Q: Cannot open ultra-high resolution images, or memory allocation limit errors appear when opening large images</summary>

This is usually not caused by the image file size on disk, but by Qt hitting its default memory allocation limit in `QImageReader` (typically around `256 MB`).

You can adjust this limit:

- **Method 1 (GUI)**: Go to `Settings` (`Ctrl+0`) > `General`, and change `Qt Image Allocation Limit`.
- **Method 2 (Config file)**: Edit `.xanylabelingrc` in your home directory and set `qt_image_allocation_limit`.
- **Method 3 (CLI argument)**: Launch with `--qt-image-allocation-limit`.

Example:
```yaml
qt_image_allocation_limit: 1024
```
```bash
xanylabeling --qt-image-allocation-limit 1024
```

- `null`: Keep default Qt limit
- `0`: Disable allocation limit completely
- Positive integer: Specify limit in MB (e.g. `512`, `1024`)

Restart the application for changes to take effect. If your computer has limited RAM, setting this to `0` is not recommended.
</details>

---

## Model & Inference Issues

<details>
<summary>Q: How to use YOLO26 with NMS (how to eliminate overlapping boxes during auto-labeling)?</summary>

YOLO26 is exported as an end-to-end (NMS-free) model by default. If you prefer NMS post-processing to eliminate duplicate overlapping boxes:

1. When exporting ONNX from PT, set `end2end=false`.
2. When `end2end=false`, post-processing is identical to YOLO11. Set **`type: yolo11`** in your custom model YAML config (do not use `type: yolo26`).

See [#1280](https://github.com/CVHub520/X-AnyLabeling/issues/1280) for details.
</details>

<details>
<summary>Q: Why does running Ultralytics (YOLO) training in precompiled executable (EXE) fail?</summary>

Running Ultralytics/YOLO model training directly within the standalone precompiled EXE is not supported. The official EXE does not bundle the full training toolchain, and frozen standalone executables cannot spawn dynamic Python training subprocesses cleanly.

Training works seamlessly when running X-AnyLabeling from a Python source environment. If you need model training, run X-AnyLabeling in a Python environment.
See [#1100](https://github.com/CVHub520/X-AnyLabeling/issues/1100).
</details>

<details>
<summary>Q: Running in venv gives: "Error loading tokenizer: No such file or directory (os error 2)"</summary>

This was caused by Python's `Traversable` resource object handling in editable installations (`pip install -e .`). This issue is resolved in recent releases by directly reading resource content via `read_text()`. If you still encounter this, pull the latest code or reinstall.
</details>

<details>
<summary>Q: Error in model predict_shapes: ModelManager.new_auto_labeling_result[AutolabelingResult].emit(): argument 1 has unexpected type 'list'</summary>

This is typically caused by loading a corrupted image file or an unsupported image format.
</details>

<details>
<summary>Q: Error in loading model: YOLOE model will not be available</summary>

See [#997](https://github.com/CVHub520/X-AnyLabeling/issues/997).
</details>

<details>
<summary>Q: Error in loading model: yoloe with error: [WinError, 1314] A required privilege is not held by the client. mobileclip_blt.pt</summary>

See [#992](https://github.com/CVHub520/X-AnyLabeling/issues/992).
</details>

<details>
<summary>Q: Error in loading custom model: Invalid config file format</summary>

See [#986](https://github.com/CVHub520/X-AnyLabeling/issues/986).
</details>

<details>
<summary>Q: Error in predict_shapes: cannot access local variable 'p' where it is not associated with a value</summary>

See [#983](https://github.com/CVHub520/X-AnyLabeling/issues/983).
</details>

<details>
<summary>Q: Error in loading model: Could not download or initialize encoder data</summary>

See [#961](https://github.com/CVHub520/X-AnyLabeling/issues/961).
</details>

<details>
<summary>Q: Where are downloaded models stored by default?</summary>

Models are saved to `~/.cache/anylabeling/` by default. See [#943](https://github.com/CVHub520/X-AnyLabeling/issues/943).
</details>

<details>
<summary>Q: Model predicted classes do not match training classes</summary>

Verify the format of your custom model config file (`*.yaml`):
1. Use 2-space indentation.
2. Keys must not contain extra spaces.
See [#923](https://github.com/CVHub520/X-AnyLabeling/issues/923).
</details>

<details>
<summary>Q: How to configure proxy for Chatbot models (e.g., Google Gemini)?</summary>

Set HTTP/HTTPS proxies in your terminal or environment configuration:
```bash
export http_proxy=http://ip:port
export https_proxy=http://ip:port
```
</details>

<details>
<summary>Q: Chatbot error: "Using SOCKS proxy, but the 'socksio' package is not installed"</summary>

When SOCKS proxy is configured in your environment (`ALL_PROXY=socks5://...`), `httpx` requires `socksio`. Install it with:
```bash
pip install "httpx[socks]"
```
</details>

<details>
<summary>Q: ERROR | model_manager:predict_shapes - Error in predict_shapes: '<=' not supported between instances of 'int' and 'str'</summary>

Check your model YAML config. See [#902](https://github.com/CVHub520/X-AnyLabeling/issues/902).
</details>

<details>
<summary>Q: OpenCV error during model prediction: (-215:Assertion failed) inv_scale_x >0 in function 'cv::resize'</summary>

- Specify image width and height explicitly in your model configuration file. See [#885](https://github.com/CVHub520/X-AnyLabeling/issues/885).
- Check whether dynamic batch was configured during export. See [#784](https://github.com/CVHub520/X-AnyLabeling/issues/784).
</details>

<details>
<summary>Q: Running yolo-pose model gives: Error in loading model: 'list' object has no attribute 'items'</summary>

The configuration file was not written according to the official template. See [#880](https://github.com/CVHub520/X-AnyLabeling/issues/880).
</details>

<details>
<summary>Q: Error in predict_shapes: list index out of range. Please check the model.</summary>

- Check if your label names are purely numeric. If so, wrap them in single quotes (e.g. `'123'`).
- Ensure the `type` field in the configuration file matches the model architecture. See [#837](https://github.com/CVHub520/X-AnyLabeling/issues/837) and [#878](https://github.com/CVHub520/X-AnyLabeling/issues/878).
</details>

<details>
<summary>Q: Error in loading model: exceptions must derive from BaseException</summary>

1. Ensure model file paths in the config file are valid and exist. See [#868](https://github.com/CVHub520/X-AnyLabeling/issues/868) and [#441](https://github.com/CVHub520/X-AnyLabeling/issues/441).
2. Check whether the ONNX model's `ir_version` is compatible with your installed ONNX runtime version. See [#1120](https://github.com/CVHub520/X-AnyLabeling/issues/1120#issuecomment-3303864917).
</details>

<details>
<summary>Q: Error in model prediction: 'int' object is not subscriptable. Please check the model.</summary>

For custom non-built-in models, verify preprocessing, inference, and postprocessing outputs. See [#828](https://github.com/CVHub520/X-AnyLabeling/issues/828).
</details>

<details>
<summary>Q: Error installing SAM 2: "from sam2 import _C"</summary>

See [#719](https://github.com/CVHub520/X-AnyLabeling/issues/719), [#842](https://github.com/CVHub520/X-AnyLabeling/issues/842), [#843](https://github.com/CVHub520/X-AnyLabeling/issues/843), [#864](https://github.com/CVHub520/X-AnyLabeling/issues/864), and [#865](https://github.com/CVHub520/X-AnyLabeling/issues/865).
</details>

<details>
<summary>Q: Downloaded models get deleted and re-downloaded every time the application launches</summary>

When existing cached models are found, the application runs an integrity check. If validation fails or files are truncated, the cached model is removed and re-downloaded.

If network issues cause download corruption, switch model download sources (GitHub or ModelScope) under User Guide settings, or deploy via [X-AnyLabeling-Server](https://github.com/CVHub520/X-AnyLabeling-Server). Also ensure model directory paths do not contain non-ASCII/Chinese characters (see [#600](https://github.com/CVHub520/X-AnyLabeling/issues/600)).
</details>

<details>
<summary>Q: How to restrict model inference to specific classes only?</summary>

For supported models (e.g. YOLO series), add `filter_classes` to your model YAML configuration. See [Custom Model Guide](https://xanylabeling.com/docs/x-anylabeling/custom_model).
</details>

<details>
<summary>Q: Error: in paint assert len(self.points) in [1, 2, 4] AssertionError</summary>

See [#491](https://github.com/CVHub520/X-AnyLabeling/issues/491).
</details>

<details>
<summary>Q: Error loading custom model</summary>

1. Confirm class names defined in the configuration match the model outputs.
2. Use [Netron](https://netron.app/) to verify input/output node names and tensor shapes match the built-in model template.
3. Check all config fields. See [#888](https://github.com/CVHub520/X-AnyLabeling/issues/888).
4. Do not modify internal built-in field names. See [#983](https://github.com/CVHub520/X-AnyLabeling/issues/983).
</details>

<details>
<summary>Q: Inference runs with no error but outputs no shapes</summary>

See [#536](https://github.com/CVHub520/X-AnyLabeling/issues/536).
</details>

<details>
<summary>Q: Grounding-DINO GPU inference error: "Expand node left operand cannot broadcast on dim 2"</summary>

See [#389](https://github.com/CVHub520/X-AnyLabeling/issues/389).
</details>

<details>
<summary>Q: Missing config: num_masks</summary>

See [#515](https://github.com/CVHub520/X-AnyLabeling/issues/515).
</details>

<details>
<summary>Q: AttributeError: 'int' object has no attribute 'replace'</summary>

Check if label names are purely numeric. All purely numeric label names must be enclosed in quotes (e.g. `'0'`, `'1'`).
</details>

<details>
<summary>Q: Unsupported model IR version: 11, max supported IR version: 10</summary>

You can downgrade the IR version of the ONNX model using Python:
```python
import onnx
onnx_model = onnx.load("/path/to/your/onnx_model")
onnx_model.ir_version = 10
onnx.save(onnx_model, "/path/to/onnx_model")
```
</details>

<details>
<summary>Q: Your model ir_version is higher than the checker's</summary>

Your installed `onnx` package is outdated. Upgrade it:
```bash
pip install --upgrade onnx
```
</details>

<details>
<summary>Q: How to use Segment Anything Video on macOS?</summary>

See [#865](https://github.com/CVHub520/X-AnyLabeling/issues/865).
</details>

<details>
<summary>Q: Running GPU version fails with: `FAIL : Failed to load library libonnxruntime_providers_cuda.so with error: libcudnn.so.9: cannot open shared object file: No such file or directory`</summary>

This indicates missing cuDNN 9 runtime libraries. You can install NVIDIA's official cuDNN wheel directly into your virtual environment without root/sudo privileges:

```bash
# For CUDA 12.x
uv pip install nvidia-cudnn-cu12
echo 'export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH' >> .venv/bin/activate
source .venv/bin/activate

# For CUDA 13.x
uv pip install nvidia-cudnn-cu13
echo 'export LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH' >> .venv/bin/activate
source .venv/bin/activate
```

For system-level cuDNN installation instructions, see [#834](https://github.com/CVHub520/X-AnyLabeling/issues/834) and [#1014](https://github.com/CVHub520/X-AnyLabeling/issues/1014).
</details>

<details>
<summary>Q: Running GPU version fails with: `ImportError: DLL load failed while importing onnx_cpp2py_export`</summary>

The installed `onnx` and `onnxruntime` versions are incompatible. See [#886](https://github.com/CVHub520/X-AnyLabeling/issues/886).
</details>

<details>
<summary>Q: Model stays in loading state indefinitely when running compiled GPU version</summary>

Caused by an incompatible combination of `onnx` and `onnxruntime-gpu`. Reference the [ONNX Runtime Compatibility Matrix](https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support) to install the matching versions.
</details>

---

## File & Export Issues

<details>
<summary>Q: YOLO-Pose exported keypoint IDs all show as 0</summary>

See [#1270](https://github.com/CVHub520/X-AnyLabeling/issues/1270).
</details>

<details>
<summary>Q: Exported mask image is completely black / empty</summary>

1. Press `Ctrl+G` to review the annotation list and ensure the annotations are `polygon` objects.
2. Ensure the export configuration file is formatted properly and that class names match the labeled class names exactly.
3. Update to the latest version or launch from source to inspect full terminal logs.
See [#1153](https://github.com/CVHub520/X-AnyLabeling/issues/1153).
</details>

<details>
<summary>Q: Precompiled version fails to extract frames when loading video files</summary>

See [#1136](https://github.com/CVHub520/X-AnyLabeling/issues/1136).
</details>

<details>
<summary>Q: Importing X-AnyLabeling labels into Labelme raises AssertionError</summary>

See [#1007](https://github.com/CVHub520/X-AnyLabeling/issues/1007).
</details>

<details>
<summary>Q: Export error: "Error occurred while exporting annotations. 'xxx' is not in list"</summary>

1. Press `Ctrl+G` to inspect all class names currently present in the annotations.
2. Check your `classes.txt` file to confirm that every annotated class name is listed in `classes.txt`.
3. Verify that you selected the matching export format (e.g., do not select segmentation/polygon export if your annotations are bounding boxes).
</details>

<details>
<summary>Q: Loading image directory causes Segmentation Fault</summary>

See [#906](https://github.com/CVHub520/X-AnyLabeling/issues/906).
</details>

<details>
<summary>Q: Uploading label file gives: "cannot identify image file xxx"</summary>

Ensure image files and label files are stored in separate directories. See [#911](https://github.com/CVHub520/X-AnyLabeling/issues/911).
</details>

<details>
<summary>Q: Loading file error: "a bytes-like object is required, not 'NoneType', ensure xxx.json is a valid label file"</summary>

Confirm that the `imagePath` field inside the `*.json` annotation matches the image file name. See [#869](https://github.com/CVHub520/X-AnyLabeling/issues/869).
</details>

<details>
<summary>Q: Imported label file appears empty</summary>

- Check if the shape type matches the export type (e.g., labeled as `rectangle` but exported as `polygon`).
- Ensure the image directory does not contain multi-level nested subdirectories. See [#839](https://github.com/CVHub520/X-AnyLabeling/issues/839).
</details>

<details>
<summary>Q: Application crashes when importing or exporting YOLO keypoint labels</summary>

Check your model YAML configuration. See [#898](https://github.com/CVHub520/X-AnyLabeling/issues/898).
</details>

<details>
<summary>Q: Error importing labels: "invalid literal for int() with base 10"</summary>

See [#782](https://github.com/CVHub520/X-AnyLabeling/issues/782).
</details>

<details>
<summary>Q: Export mask error: "imageWidth"</summary>

See [#477](https://github.com/CVHub520/X-AnyLabeling/issues/477).
</details>

<details>
<summary>Q: "operands could not be broadcast together with shapes (0,) (1,2)"</summary>

See [#492](https://github.com/CVHub520/X-AnyLabeling/issues/492).
</details>

<details>
<summary>Q: Exporting keypoint labels error: "int0 argument must be a string, a bytes-like object or a number, not 'NoneType'"</summary>

Missing `group_id`. Ensure every bounding box and its associated keypoints share a matching group ID.
</details>

<details>
<summary>Q: "'NoneType' object has no attribute 'shape'"</summary>

Check whether the file path contains Chinese characters or non-ASCII characters. Move the files to a standard path without special characters.
</details>
