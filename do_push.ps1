# ComfyUI-NunchakuFluxLoraStacker: add / commit / push (サンドボックス外実行用)
# 実行: powershell -ExecutionPolicy Bypass -File "d:\USERFILES\GitHub\ComfyUI-NunchakuFluxLoraStacker\do_push.ps1"

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# ロックファイル・残骸を削除
Remove-Item -Force .git\index.lock -ErrorAction SilentlyContinue
Get-ChildItem -Path .git\objects -Recurse -Filter "tmp_obj_*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

# プロキシを無効化（gh や curl を使う場合用）
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:http_proxy = ""
$env:https_proxy = ""
$env:ALL_PROXY = ""

# 変更をステージ（md/ は .gitignore 済み）
git add .gitignore __init__.py pyproject.toml
git add AILab_SAM3Segment.py js/colorWidget.js nodes/load_image_ussoewwin.py nodes/lora_analyzer_node.py png/loraana.png
git add sam3/

git status
$msg = Read-Host "Commit message (Enter = default)"
if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "Add Load Image node, LoRA Analyzer, SAM3; update .gitignore (md/)" }
git commit -m $msg
git push
