import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def plot_multiple_pae_heatmaps_with_confidence(json_files, confidence_values,
                               labels=None,
                               cmap='Purples_r',
                               vmin=0, 
                               vmax=30,
                               figsize_per_plot=(6, 5),
                               save_path=None,
                               dpi=150):
    """
    绘制多张 PAE 热图拼接图，每张图固定每行 4 个子图。
    """
    n = len(json_files)
    if n == 0:
        raise ValueError("No JSON files provided.")
    
    ncols = 4
    nrows = int(np.ceil(n / ncols))  # 计算需要的行数
    
    # 固定每行4列的布局，使用固定的子图大小
    fig_width = figsize_per_plot[0] * ncols
    fig_height = figsize_per_plot[1] * nrows + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, json_file in enumerate(json_files):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        pae = np.array(data['pae'])
        
        im = axes[i].imshow(pae, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper', aspect='auto')
        formatted_label = format_label(labels[i] if labels else os.path.basename(json_file))
        axes[i].set_title(formatted_label, fontsize=12)
        
        if json_file in confidence_values:
            conf_info = confidence_values[json_file]
            iptm_val = conf_info.get('iptm', conf_info.get('ipTM', 'N/A'))
            ptm_val = conf_info.get('ptm', conf_info.get('pTM', 'N/A'))
            axes[i].text(0.5, -0.15, f"ipTM: {iptm_val:.3f}, pTM: {ptm_val:.3f}", 
                         transform=axes[i].transAxes, ha="center", va="top",
                         fontsize=10)
        
        axes[i].set_xlabel('Scored Residue')
        axes[i].set_ylabel('Aligned Residue')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    cbar_ax = fig.add_axes([0.1, 0.05, 0.8, 0.02])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('PAE (Predicted Aligned Error)', fontsize=12)

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()

def format_label(label):
    """
    将标签的大写化，并将下划线 '_' 替换为连接符 '-'，同时保持 A-B-C 的结构。
    """
    # 将整个标签转换为大写
    label_upper = label.upper()
    
    # 将下划线替换为连接符
    formatted_label = label_upper.replace('_', '-')
    
    return formatted_label

# 示例使用
labels = ['flg22_fls2_example', 'test_label', 'ANOTHER_TEST_LABEL']

formatted_labels = [format_label(label) for label in labels]

for original, formatted in zip(labels, formatted_labels):
    print(f"Original: {original} -> Formatted: {formatted}")

# 新的主程序：按照指定数量分批生成图像
root_directory = ""
json_files = sorted(glob.glob(os.path.join(root_directory, "**/*data_0.json"), recursive=True))

confidence_values = {}
for file_path in json_files:
    dir_path = os.path.dirname(file_path)
    conf_files = glob.glob(os.path.join(dir_path, "*_confidences_0.json"))
    if conf_files:
        with open(conf_files[0], 'r') as f:
            conf_data = json.load(f)
            confidence_values[file_path] = {
                'iptm': conf_data.get('iptm', conf_data.get('ipTM', 0.0)),
                'ptm': conf_data.get('ptm', conf_data.get('pTM', 0.0))
            }
    else:
        confidence_values[file_path] = {'iptm': 0.0, 'ptm': 0.0}

labels = []
for file_path in json_files:
    parent_dir = os.path.basename(os.path.dirname(file_path))
    formatted_label = format_label(parent_dir)
    labels.append(formatted_label)

# 按照指定的数量分批处理
batches = [(0, 28), (28, 56), (56, 81)]
for start, end in batches:
    batch_json = json_files[start:end]
    batch_labels = labels[start:end]
    batch_conf = {k: confidence_values[k] for k in batch_json}
    
    part_num = batches.index((start, end)) + 1
    save_name = f"combined_pae_heatmaps_part{part_num}.png"
    
    plot_multiple_pae_heatmaps_with_confidence(
        json_files=batch_json,
        confidence_values=batch_conf,
        labels=batch_labels,
        cmap='Purples_r',
        vmin=0,
        vmax=30,
        figsize_per_plot=(6, 5),  # 固定子图大小
        save_path=save_name,
        dpi=150
    )




