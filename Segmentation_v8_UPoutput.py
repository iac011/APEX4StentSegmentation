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

    def compute_encapsulation(self, stent_mask, calc_mask, dilation_iter=5):
        kernel = np.ones((3, 3), np.uint8)

        # 1. 支架膨胀
        stent_dilated = cv2.dilate(stent_mask, kernel, iterations=dilation_iter)

        # 2. 包裹区域（钙化贴着支架）
        encap_mask = ((calc_mask > 0) & (stent_dilated > 0)).astype(np.uint8) * 255

        # 3. 面积
        encap_area = int(np.sum(encap_mask > 0))
        calc_area = int(np.sum(calc_mask > 0))
        stent_area = int(np.sum(stent_mask > 0))

        # 4. 比例（两个都很有用）
        encap_ratio_calc = encap_area / (calc_area + 1e-6)
        encap_ratio_stent = encap_area / (stent_area + 1e-6)

        return encap_mask, encap_area, encap_ratio_calc, encap_ratio_stent

    def compute_spatial_distribution(self, stent_mask, calc_mask):
        # 距离场（到支架）
        dist_map = cv2.distanceTransform((stent_mask == 0).astype(np.uint8), cv2.DIST_L2, 5)

        # 钙化区域距离
        calc_distances = dist_map[calc_mask > 0]
        hist, bins = np.histogram(calc_distances, bins=20)
        if len(calc_distances) == 0:
            return 0, 0, 0

        mean_dist = float(np.mean(calc_distances))
        max_dist = float(np.max(calc_distances))
        min_dist = float(np.min(calc_distances))

        return mean_dist, max_dist, min_dist

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

# ===针对性优化方案=== 首先根据参考尺寸，精确清洗 Legend 和 Scale bar
# 为了确保能抓取到这些微弱的点，我们需要做两件事：
# 取消固定阈值：改用“非零判定”，即只要图中存在比背景噪声亮的像素，就认为是信号 。
# 材质判定修正：你的 Mosaic element_BSE.png 中支架呈黑色孔洞状（由于是血管横截面），而支架杆（Rod）则隐藏在特定的元素通道中 。

    def clean_eds_artifacts(self, image):  # 添加 self，使其成为类方法
        """
        根据参考尺寸，精确清洗 Legend 和 Scale bar
        """
        h, w = image.shape[:2]
        # 清洗左侧 Legend
        x_end = int(2.2 / 20 * w)
        y_start, y_end = int(12 / 15 * h), int(13.5 / 15 * h)
        image[y_start:y_end, 0:x_end] = 0

        # 清洗底部 Scale bar
        scale_y_start = int((15 - 1) / 15 * h)
        image[scale_y_start:, :] = 0
        return image

    def analyze(self, prefix):
        ch = {}
        elems = ["BSE", "CA", "P", "CO", "CR", "FE", "PT", "NI", "MO", "W"]

        for e in elems:
            # 支持更多格式匹配
            match = [f for f in self.input_dir.iterdir() if prefix in f.name.upper() and f"_{e}" in f.name.upper()]
            if match:
                img = cv2.imread(str(match[0]), 0)
                img = self.clean_eds_artifacts(img)
                ch[e] = img.astype(np.float32) / 255.0
            else:
                ch[e] = None

        if ch.get("BSE") is None: return None

        # --- 1. 支架提取逻辑 (Stent: Co, Cr, Fe, Pt) ---
        # 提高支架阈值，支架信号通常较强，0.15-0.2 可以过滤掉大部分背景随机噪声
        stent_logic = np.zeros_like(ch["BSE"], dtype=bool)
        for e in ["CO", "CR", "FE", "PT"]:
            if ch[e] is not None:
                stent_logic |= (ch[e] > 0.02)  # 提高门限

        # 转换为二值图进行面积过滤
        stent_mask = (stent_logic.astype(np.uint8)) * 255

        # 【新增：面积过滤】 杀掉所有背景里的“碎点”噪声
        # 支架杆截面通常很大，这里设定最小面积为 10 像素 (不宜过大>20，因支架杆本身就小)
        stent_mask = self.area_filter(stent_mask, min_area=5)

        # --- 2. 钙化提取逻辑 (Calc: Ca, P) ---
        # 钙化信号可能较弱，保持 0.02 阈值，但通过逻辑剔除背景
        calc_logic = np.zeros_like(ch["BSE"], dtype=bool)
        if ch.get("CA") is not None:
            calc_logic |= (ch["CA"] > 0.02)
        if ch.get("P") is not None:
            calc_logic |= (ch["P"] > 0.02)

        # 【核心优化：物理排他】
        # 1. 钙化必须出现在 BSE 衬度较低（灰度值>50）的区域，排除极黑的背景（树脂）
        # 2. 钙化区域必须剔除已经判定的支架区域，防止红蓝混淆
        bse_valid = (ch["BSE"] > (50 / 255.0))
        calc_logic &= bse_valid
        calc_logic &= (~(stent_mask > 0))  # 强制排除支架像素

        calc_mask = (calc_logic.astype(np.uint8)) * 255
        # 钙化通常呈层状或团块状，过滤面积小于 150 的噪点
        calc_mask = self.area_filter(calc_mask, min_area=150)

        # --- 3. 形态学精修 ---
        kernel = np.ones((3, 3), np.uint8)
        # 闭运算：先膨胀后腐蚀，填补碎点间的空洞
        stent_mask = cv2.morphologyEx(stent_mask, cv2.MORPH_CLOSE, kernel)
        calc_mask = cv2.morphologyEx(calc_mask, cv2.MORPH_CLOSE, kernel)

        # === 新增分析 ===
        encap_mask, encap_area, encap_ratio_calc, encap_ratio_stent = \
            self.compute_encapsulation(stent_mask, calc_mask)

        mean_dist, max_dist, min_dist = \
            self.compute_spatial_distribution(stent_mask, calc_mask)

        # --- 4. 保存 ---
        stent_save_path = self.output_dir / f"{prefix}_STENT_LABEL.png"
        calc_save_path = self.output_dir / f"{prefix}_CALC_LABEL.png"
        cv2.imwrite(str(stent_save_path), stent_mask)
        cv2.imwrite(str(calc_save_path), calc_mask)

        # --- 5. 材质判定 ---
        # === Material Classification (Upgraded) ===

        material = "Unknown"

        # 提取元素强度（用 max 或 mean 都可以，建议 max）
        def get_signal(e):
            return float(np.max(ch[e])) if ch.get(e) is not None else 0.0

        signals = {
            "PT": get_signal("PT"),
            "FE": get_signal("FE"),
            "CR": get_signal("CR"),
            "CO": get_signal("CO"),
            "NI": get_signal("NI"),
            "MO": get_signal("MO"),
            "W": get_signal("W")
        }

        # 阈值（可调）
        T = 0.02

        # === 1. Pt-based 优先 ===
        if signals["PT"] > T:
            if signals["CO"] > T:
                material = "Pt(core)/MP35N(shell)"
            else:
                material = "PtCr"

        # === 2. W → L605（强特征）===
        elif signals["W"] > T:
            material = "L605"

        # === 3. Co-based alloys ===
        elif signals["CO"] > T:
            if signals["MO"] > T and signals["NI"] > T:
                material = "MP35N"
            elif signals["CR"] > T:
                material = "L605"
            else:
                material = "Co-based Alloy"

        # === 4. Fe-based ===
        elif signals["FE"] > T:
            if signals["CR"] > T and signals["NI"] > T:
                material = "316L"
            else:
                material = "Fe-based"

        stent_area = int(np.sum(stent_mask > 0))
        calc_area = int(np.sum(calc_mask > 0))


        # 6. return ouptput CSV
        return {
            "Sample": prefix,
            "Stent Material": material,

            # 基础面积
            "Stent_Rod_Area": stent_area,
            "Calc_Area": calc_area,

            # 包裹
            "Encap_Area": encap_area,
            "Encap_Ratio_vs_Calc": encap_ratio_calc,
            "Encap_Ratio_vs_Stent": encap_ratio_stent,

            # 空间分布
            "Calc_Dist_Mean": mean_dist,
            "Calc_Dist_Max": max_dist,
            "Calc_Dist_Min": min_dist
            }


    def area_filter(self, mask, min_area):
        """
        工具函数：滤除二值图中面积小于 min_area 的孤立区域
        """
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        new_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                new_mask[labels == i] = 255
        return new_mask

if __name__ == "__main__":
    expert = StentDiagnosticExpert(input_dir="MosaicRaw_data", output_dir="./Mosaic_results")
    expert.run()

