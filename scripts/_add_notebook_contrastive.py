"""Thêm mục contrastive (E3--E6) vào notebook M8."""

from __future__ import annotations

from pathlib import Path

import nbformat

NB = Path(r"D:\IOT\notebooks\M8_CIFAR10_MultiTask.ipynb")


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


NEW_CELLS = [
    md(
        "---\n\n"
        "## 10. Mở rộng contrastive: NT-Xent và SupCon (E3–E6)\n\n"
        "Phần này triển khai bảng 5 của đề cương. Hai hàm mất mát được cài đúng theo tài liệu gốc:\n\n"
        "$$L_{\\mathrm{NT\\text{-}Xent}} = \\frac{1}{2N}\\sum_i -\\log"
        "\\frac{\\exp(s_{i,j(i)}/\\tau)}{\\sum_{k\\neq i}\\exp(s_{i,k}/\\tau)}$$\n\n"
        "$$L_{\\mathrm{SupCon}} = \\sum_i \\frac{-1}{|A(i)|}\\sum_{a\\in A(i)}\\log"
        "\\frac{\\exp(s_{i,a}/\\tau)}{\\sum_{k\\neq i}\\exp(s_{i,k}/\\tau)},\\quad"
        "A(i)=\\{a\\neq i: \\tilde y_a = \\tilde y_i\\}$$\n\n"
        "**Ràng buộc bắt buộc:** SupCon chỉ dùng $D_L$ (không đọc nhãn thật của $D_U$);\n"
        "NT-Xent dùng $D_{train}$ vì không cần nhãn; Projection Head **chỉ** dùng khi huấn luyện\n"
        "nên số tham số suy luận không đổi."
    ),
    code(
        "from m8.contrastive import contrastive_loss, nt_xent_loss, supcon_loss, ProjectionHead\n"
        "\n"
        "# --- ví dụ tính tay cho NT-Xent ---\n"
        "z = torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., 1.]])\n"
        "import math\n"
        "got = nt_xent_loss(z, temperature=0.5).item()\n"
        "expect = math.log(math.exp(2.0) + 2.0) - 2.0\n"
        "print(f'NT-Xent: nhận {got:.6f} | công thức tay {expect:.6f} -> '\n"
        "      f\"{'ĐẠT' if abs(got - expect) < 1e-6 else 'SAI'}\")\n"
        "\n"
        "# --- ví dụ tính tay cho SupCon ---\n"
        "z2 = torch.tensor([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])\n"
        "lab = torch.tensor([0, 0, 1, 1])\n"
        "l_sup = supcon_loss(z2, lab, temperature=0.5)\n"
        "zn = torch.nn.functional.normalize(z2, dim=1); sim = zn @ zn.t() / 0.5\n"
        "manual = 0.0\n"
        "for i in range(4):\n"
        "    others = [k for k in range(4) if k != i]\n"
        "    lse = torch.logsumexp(sim[i, others], dim=0)\n"
        "    pos = [k for k in others if lab[k] == lab[i]]\n"
        "    manual += float((-torch.stack([sim[i, p] for p in pos]) + lse).mean())\n"
        "manual /= 4\n"
        "print(f'SupCon : nhận {l_sup.item():.6f} | công thức tay {manual:.6f} -> '\n"
        "      f\"{'ĐẠT' if abs(l_sup.item() - manual) < 1e-5 else 'SAI'}\")\n"
        "print('SupCon không có positive pair ->', supcon_loss(z2, torch.arange(4)).item(), '(phải là 0)')"
    ),
    code(
        "# --- Projection Head không ảnh hưởng suy luận ---\n"
        "m_plain = build_model(with_rotation_head=True, seed=0)\n"
        "m_proj = build_model(with_rotation_head=True, proj_dim=C.PROJ_DIM, seed=0)\n"
        "s_plain, s_proj = model_summary(m_plain), model_summary(m_proj)\n"
        "print('không có Projection Head: suy luận', f\"{s_plain['inference_total']:,}\", 'tham số')\n"
        "print('có Projection Head      : suy luận', f\"{s_proj['inference_total']:,}\", 'tham số |',\n"
        "      'Projection Head', f\"{s_proj.get('projection_head', 0):,}\", 'tham số')\n"
        "assert s_proj['inference_total'] == s_plain['inference_total'] == 11_169_858\n"
        "print('=> số tham số suy luận KHÔNG đổi: ĐẠT')"
    ),
    code(
        "# --- hai view contrastive và dataset không mang nhãn ---\n"
        "from m8.data import ContrastiveViewDataset\n"
        "\n"
        "idx_sup = bench.dl(42, 0.10)\n"
        "ds_sup = ContrastiveViewDataset(train_base, idx_sup, return_superclass=True)\n"
        "a, b, y, _ = ds_sup[0]\n"
        "print('SupCon  : view1', tuple(a.shape), '| view2', tuple(b.shape), '| nhãn', y,\n"
        "      '| hai view khác nhau:', not torch.allclose(a, b))\n"
        "\n"
        "ds_unsup = ContrastiveViewDataset(train_base, bench.train_idx, return_superclass=False)\n"
        "_, _, y2, _ = ds_unsup[0]\n"
        "print('NT-Xent : nhãn trả về', y2, '(= -1 nghĩa là dataset không mang nhãn -> không rò rỉ)')\n"
        "\n"
        "fig, axes = plt.subplots(1, 4, figsize=(10, 2.7))\n"
        "def _disp(t):\n"
        "    return np.clip(t.detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)\n"
        "for ax, (im, title) in zip(axes, [(np.asarray(train_base[int(idx_sup[0])][0]), 'ảnh gốc'),\n"
        "                                  (_disp(a), 'contrastive view 1'),\n"
        "                                  (_disp(b), 'contrastive view 2')]):\n"
        "    ax.imshow(im); ax.set_title(title, fontsize=9); ax.axis('off')\n"
        "from m8.data import RotationViewDataset as _RVD\n"
        "import numpy as np\n"
        "axes[3].imshow(np.rot90(np.asarray(train_base[int(idx_sup[0])][0]), k=1))\n"
        "axes[3].set_title('nhánh xoay k=1', fontsize=9); axes[3].axis('off')\n"
        "plt.tight_layout(); plt.show()"
    ),
    code(
        "# --- kết quả khối mở rộng đã chạy (nếu có) ---\n"
        "from m8.config import CHECKPOINT_EXT_DIR\n"
        "from m8.experiments import load_all_runs_multi, summarize_runs as _summ\n"
        "\n"
        "runs_ext = load_all_runs(CHECKPOINT_EXT_DIR)\n"
        "print('số lượt mở rộng đã có:', len(runs_ext))\n"
        "if runs_ext:\n"
        "    summary_all = _summ(load_all_runs_multi(ROOT / 'results' / 'checkpoints', CHECKPOINT_EXT_DIR))\n"
        "    rows = []\n"
        "    for cfg in ('E1', 'E2-L', 'E2', 'E3', 'E4', 'E5', 'E6'):\n"
        "        for pct in sorted({r['label_pct'] for r in summary_all['runs']}):\n"
        "            a = summary_all['aggregate'].get(f'{cfg}_{pct}')\n"
        "            d = summary_all['paired_delta'].get(f'{cfg}_minus_E1_{pct}')\n"
        "            if not a:\n"
        "                continue\n"
        "            rows.append({'Cấu hình': cfg, 'Nhãn': f'{pct}%',\n"
        "                         'Test Macro-F1': f\"{a['test_macro_f1']['mean']:.4f} ± {a['test_macro_f1']['std']:.4f}\",\n"
        "                         'Δ so E1 (điểm %)': ('—' if cfg == 'E1' or not d else f\"{d['mean']:+.2f}\"),\n"
        "                         'số lần dương': ('—' if cfg == 'E1' or not d else\n"
        "                                         f\"{sum(1 for v in d['deltas_pp'] if v > 0)}/{d['n']}\")})\n"
        "    display(pd.DataFrame(rows))\n"
        "else:\n"
        "    print('Chưa có kết quả mở rộng — chạy scripts/run_extensions.py trước.')"
    ),
    code(
        "# --- khảo sát λ_c: CHỈ dùng validation, không đụng tới test ---\n"
        "import json\n"
        "sweep_path = ROOT / 'results' / 'lambda_c_sweep.json'\n"
        "if sweep_path.exists():\n"
        "    sw = json.loads(sweep_path.read_text(encoding='utf-8'))\n"
        "    rec = pd.DataFrame(sw['records'])[['config', 'kind', 'source', 'label_pct',\n"
        "                                       'lambda_c', 'best_val_macro_f1', 'best_epoch']]\n"
        "    display(rec.sort_values(['config', 'lambda_c']).reset_index(drop=True))\n"
        "    print('λ_c được chọn theo validation:', json.dumps(sw['chosen'], ensure_ascii=False))\n"
        "else:\n"
        "    print('Chưa có khảo sát — chạy scripts/sweep_lambda_c.py')"
    ),
    md(
        "**Kết luận của phần mở rộng** được sinh tự động từ số liệu (xem ô dưới), để tránh việc\n"
        "kết luận đi trước kết quả."
    ),
    code(
        "from make_report_data import _extension_text\n"
        "\n"
        "if runs_ext:\n"
        "    analysis, conclusion = _extension_text(summary_all)\n"
        "    print('PHÂN TÍCH:\\n', analysis.replace('\\\\cfg{', '').replace('}', '').replace('\\\\%', '%'))\n"
        "    print()\n"
        "    print('KẾT LUẬN:\\n', conclusion.replace('\\\\cfg{', '').replace('}', '')\n"
        "          .replace('\\\\%', '%').replace('\\\\textbf{', '').replace('\\\\,', ' '))\n"
        "else:\n"
        "    print('Chưa có kết quả mở rộng.')"
    ),
]


def main() -> int:
    nb = nbformat.read(NB, as_version=4)
    # chèn trước mục "## 10. Ghi chú về tái lập và giới hạn"
    insert_at = None
    for i, c in enumerate(nb.cells):
        if c.cell_type == "markdown" and "Ghi chú về tái lập" in c.source:
            insert_at = i
            break
    if insert_at is None:
        print("không tìm thấy mục chèn")
        return 1
    # tránh chèn trùng nếu chạy lại
    if any(c.cell_type == "markdown" and "Mở rộng contrastive" in c.source for c in nb.cells):
        print("đã có mục mở rộng, bỏ qua")
        return 0
    nb.cells[insert_at:insert_at] = NEW_CELLS
    nbformat.write(nb, NB)
    print(f"đã chèn {len(NEW_CELLS)} ô vào vị trí {insert_at}; tổng {len(nb.cells)} ô")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
