#!/usr/bin/env bash
# 下载 LLM_claude 项目相关论文 PDF
# 使用方式：bash download_papers.sh
# 需要在可访问 arxiv.org 的网络环境中运行

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS_DIR="$SCRIPT_DIR/papers"
mkdir -p "$PAPERS_DIR"

download_paper() {
  local arxiv_id="$1"
  local filename="$2"
  local title="$3"
  local dest="$PAPERS_DIR/$filename"

  if [ -f "$dest" ] && [ "$(wc -c < "$dest")" -gt 10000 ]; then
    echo "[跳过] $title（已存在）"
    return 0
  fi

  echo "[下载] $title ..."
  if curl -L --max-time 60 --retry 3 \
       -H "User-Agent: Mozilla/5.0" \
       -o "$dest" \
       "https://arxiv.org/pdf/${arxiv_id}"; then
    SIZE=$(wc -c < "$dest")
    if [ "$SIZE" -gt 10000 ]; then
      echo "[完成] $filename (${SIZE} bytes)"
    else
      echo "[失败] $filename 文件过小，可能下载出错"
      rm -f "$dest"
    fi
  else
    echo "[失败] $title 下载失败"
  fi
  sleep 1  # 避免请求过于频繁
}

echo "===== 开始下载论文 PDF ====="
echo "目标目录：$PAPERS_DIR"
echo ""

# Transformer 基础
download_paper "1706.03762" "attention-is-all-you-need.pdf"          "Attention Is All You Need"
download_paper "1810.04805" "bert-1810.04805.pdf"                     "BERT"
download_paper "2005.14165" "gpt3-2005.14165.pdf"                     "GPT-3"

# 推理优化
download_paper "2205.14135" "flashattention-2205.14135.pdf"           "FlashAttention"
download_paper "2307.08691" "flashattention2-2307.08691.pdf"          "FlashAttention-2"
download_paper "2309.06180" "pagedattention-2309.06180.pdf"           "PagedAttention (vLLM)"
download_paper "2305.13245" "gqa-2305.13245.pdf"                      "GQA"

# DeepSeek 系列
download_paper "2412.19437" "deepseek-v3-2412.19437.pdf"              "DeepSeek-V3"
download_paper "2602.21548" "dualpath-2602.21548.pdf"                 "DualPath"

# PD 分离 & Agentic 推理
download_paper "2403.02310" "sarathi-serve-2403.02310.pdf"            "Sarathi-Serve"
download_paper "2407.00079" "mooncake-2407.00079.pdf"                 "Mooncake"

echo ""
echo "===== 下载完成 ====="
ls -lh "$PAPERS_DIR"
