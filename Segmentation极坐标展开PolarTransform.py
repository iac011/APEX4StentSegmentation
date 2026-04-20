import cv2
import numpy as np
from pathlib import Path

# ===极坐标展开图===
# 为什么这张图对您的论文至关重要？
# 在传统的圆形截图中，很难量化“钙化是不均匀的”这一结论。通过这张展开图，您可以进行以下深度分析：
# 钙化厚度定量分析 (Thickness Profiling)：
# 展开后的红色区域（钙化）在 Y 轴上的高度即代表了该角度下的厚度。您可以绘制一条“厚度-角度”曲线，证明钙化在某些支架杆周围存在局部加强。
# 支架嵌入深度 (Embedding Depth)：
# 观察蓝色点（支架杆）在 Y 轴上的位置。如果蓝色点完全没入红色区域，说明支架被钙化完全包裹；如果蓝色点在红色区域上方，说明钙化发生在支架内侧。
# 识别局部降解风险：
# 如果某一角度范围（X 轴段）缺失了蓝色点（支架信号），但 BSE 图像显示有结构空洞，这便是潜在的腐蚀或断裂证据。
#
# # 在论文的方法论部分，这样描述这套系统：
# "To quantitatively evaluate the spatial relationship between the stent struts and calcified plaque, a custom-built AI pipeline performed a unbiased elemental-driven segmentation. The circular cross-sections were subsequently flattened via a Cartesian-to-Polar transformation, allowing for the high-resolution mapping of calcification thickness and strut embedding depth across the entire 360° vessel circumference."
# 生成的 _POLAR_OVERLAY.png 应该是一张长方形的图片，左侧对应 0°，右侧对应 360°。

def generate_polar_figures(prefix, input_dir, output_dir, resolution=(1200, 600)):
    """
    将圆形横截面展开为矩形线性图
    resolution: (角度分辨率, 径向分辨率)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载原始 BSE 和 之前生成的 Mask
    bse = cv2.imread(str(input_path / f"{prefix}_BSE.png"), 0)
    stent = cv2.imread(str(output_path / f"{prefix}_STENT_LABEL.png"), 0)
    calc = cv2.imread(str(output_path / f"{prefix}_CALC_LABEL.png"), 0)  # 假设这是之前生成的钙化掩模

    if bse is None or stent is None:
        print(f"跳过 {prefix}: 找不到必要文件")
        return

    # 2. 自动计算血管中心 (利用支架杆分布的质心)
    M = cv2.moments(stent)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = bse.shape[1] // 2, bse.shape[0] // 2

    center = (cX, cY)
    # 计算最大半径 (支撑整个图像范围)
    max_radius = int(np.sqrt(cX ** 2 + cY ** 2))

    # 3. 执行极坐标转换 (Warp Polar)
    # 这里的目标是：横轴代表角度 (0-360°)，纵轴代表从中心向外的距离
    flags = cv2.WARP_POLAR_LINEAR + cv2.INTER_CUBIC

    # 转换函数
    polar_bse = cv2.warpPolar(bse, resolution, center, max_radius, flags)
    polar_stent = cv2.warpPolar(stent, resolution, center, max_radius, flags)
    polar_calc = cv2.warpPolar(calc, resolution, center, max_radius, flags)

    # 4. 旋转图像使之符合论文习惯：X轴为角度，Y轴为半径
    # cv2.warpPolar 默认输出中，第一维是半径，第二维是角度
    # 我们旋转后，图像顶部为血管中心，底部为血管外壁
    polar_bse = cv2.rotate(polar_bse, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar_stent = cv2.rotate(polar_stent, cv2.ROTATE_90_COUNTERCLOCKWISE)
    polar_calc = cv2.rotate(polar_calc, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # 5. 生成伪彩色叠加图 (用于展示空间关系)
    # 蓝色=支架, 红色=钙化
    overlay = cv2.cvtColor(polar_bse, cv2.COLOR_GRAY2BGR)
    overlay[polar_stent > 127] = [255, 0, 0]  # Blue
    overlay[polar_calc > 127] = [0, 0, 255]  # Red

    # 保存结果
    cv2.imwrite(str(output_path / f"{prefix}_POLAR_BSE.png"), polar_bse)
    cv2.imwrite(str(output_path / f"{prefix}_POLAR_OVERLAY.png"), overlay)

    print(f"成功生成 {prefix} 的极坐标展开图")


# 示例调用
if __name__ == "__main__":
    # 请替换为您实际的文件夹路径
    generate_polar_figures("MOSAIC ELEMENT", "./raw_data", "./results")