import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import os


class StentDiagnosticExpert:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 针对稀疏 EDS 信号，调低阈值到 0.02 (约 5/255)
        # 只要有一点点亮色就被捕捉
        self.T = {
            "Co": 0.02, "Cr": 0.02, "Fe": 0.02, "Pt": 0.02,
            "Ni": 0.02, "Mo": 0.02, "W": 0.02, "Ca": 0.02, "P": 0.02
        }

    def run(self):
        print(f"--- 诊断开始 ---")
        print(f"正在检查目录: {self.input_dir.absolute()}")

        # 检查目录下所有文件
        all_files = list(self.input_dir.iterdir())
        print(f"目录下总文件数: {len(all_files)}")
        if len(all_files) > 0:
            print(f"前 5 个文件名示例: {[f.name for f in all_files[:5]]}")

        # 尝试匹配 BSE 文件 (不区分大小写，支持 jpg/png/jpeg)
        prefixes = []
        for f in all_files:
            if "_BSE" in f.name.upper():
                # 提取前缀
                prefix = f.name.upper().split("_BSE")[0]
                prefixes.append(prefix)

        prefixes = sorted(list(set(prefixes)))
        print(f"成功识别的样本前缀: {prefixes}")

        if not prefixes:
            print("错误：没有找到包含 '_BSE' 字样的图像文件！请检查文件名。")
            return

        data = []
        for p in prefixes:
            print(f">>> 正在处理样本: {p}")
            res = self.analyze(p)
            if res:
                data.append(res)

        if data:
            df = pd.DataFrame(data)
            csv_path = self.output_dir / "Stent_Material_Report.csv"
            df.to_csv(csv_path, index=False)
            print(f"--- 处理完成！CSV 已保存至: {csv_path} ---")
        else:
            print("警告：虽然找到了文件，但 analyze 函数未返回任何数据。")

# ===针对性优化方案===
# 为了确保能抓取到这些微弱的点，我们需要做两件事：
# 取消固定阈值：改用“非零判定”，即只要图中存在比背景噪声亮的像素，就认为是信号 。
# 材质判定修正：你的 Mosaic element_BSE.png 中支架呈黑色孔洞状（由于是血管横截面），而支架杆（Rod）则隐藏在特定的元素通道中 。

    def analyze(self, prefix):
        ch = {}
        elems = ["BSE", "CA", "P", "CO", "CR", "FE", "PT", "NI", "MO", "W"]

        for e in elems:
            match = [f for f in self.input_dir.iterdir() if prefix in f.name.upper() and f"_{e}" in f.name.upper()]
            if match:
                img = cv2.imread(str(match[0]), 0)
                ch[e] = img.astype(np.float32) / 255.0
            else:
                ch[e] = None

        if ch.get("BSE") is None: return None

        # --- 1. 提取支架信号 (金属通道并集) ---
        # 适当提高支架阈值 (0.05)，防止背景噪点把钙化带进去
        stent_logic = np.zeros_like(ch["BSE"], dtype=bool)
        for e in ["CO", "CR", "FE", "PT"]:
            if ch[e] is not None:
                stent_logic |= (ch[e] > 0.05)

        stent_mask = (stent_logic.astype(np.uint8)) * 255

        # --- 2. 提取钙化信号 (Ca+P 并集，且必须排除支架区域) ---
        # 核心逻辑：(Ca > T 或 P > T) 且 (支架强度极低)
        calc_logic = np.zeros_like(ch["BSE"], dtype=bool)
        if ch.get("CA") is not None:
            calc_logic |= (ch["CA"] > 0.05)
        if ch.get("P") is not None:
            calc_logic |= (ch["P"] > 0.05)

        # 物理排他：如果一个像素已经是支架了，就不能是钙化
        calc_logic &= (~stent_logic)
        calc_mask = (calc_logic.astype(np.uint8)) * 255

        # --- 3. 形态学优化 (防止点状信号丢失) ---
        kernel = np.ones((5, 5), np.uint8)
        # 支架进行膨胀和闭运算
        stent_mask = cv2.morphologyEx(stent_mask, cv2.MORPH_CLOSE, kernel)
        stent_mask = cv2.dilate(stent_mask, kernel, iterations=1)

        # 钙化进行平滑处理
        calc_mask = cv2.morphologyEx(calc_mask, cv2.MORPH_CLOSE, kernel)

        # --- 4. 保存两个独立的 Mask 文件 ---
        # 确保这里有两个保存语句！
        stent_save_path = self.output_dir / f"{prefix}_STENT_LABEL.png"
        calc_save_path = self.output_dir / f"{prefix}_CALC_LABEL.png"

        cv2.imwrite(str(stent_save_path), stent_mask)
        cv2.imwrite(str(calc_save_path), calc_mask)

        print(f"    [*] 成功保存独立掩模: Stent -> {stent_save_path.name}, Calc -> {calc_save_path.name}")

        # --- 5. 材质判定 ---
        material = "Unknown"
        metal_scores = {e: np.max(ch[e]) for e in ["CO", "FE", "PT"] if ch[e] is not None}
        if metal_scores:
            best_metal = max(metal_scores, key=metal_scores.get)
            if best_metal == "PT":
                material = "PtCr"
            elif best_metal == "CO":
                material = "MP35N" if (ch.get("NI") is not None and np.max(ch["NI"]) > 0.1) else "L605"
            elif best_metal == "FE":
                material = "316L"

        return {
            "Sample": prefix,
            "Material": material,
            "Stent_Rod_Count": cv2.connectedComponents(stent_mask)[0] - 1,
            "Calc_Area_Ratio": np.sum(calc_mask > 0) / calc_mask.size
        }

# 修改这里：如果你的图片就在脚本旁边，把 "./raw_data" 改成 "."
if __name__ == "__main__":
    expert = StentDiagnosticExpert(input_dir="./raw_data", output_dir="./results")
    expert.run()