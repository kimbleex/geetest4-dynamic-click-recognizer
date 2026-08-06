最近在一个浏览器自动化项目中遇到了一类 GeeTest4 动态点选验证码：页面上方给出一个目标图标，下方的背景区域里排列着 4 个候选图标，需要找出与目标一致的那一个并点击。

这类图片不适合直接用固定坐标，因为**候选项的位置和内容会动态变化**；。

思路是：先找到 4 个候选区域，再提取目标和候选项的前景形状，最后综合多种相似度指标选出最佳结果。

基于 `OpenCV + NumPy` 实现了一个方案，它最终返回候选项中心点坐标，可以直接交给 `Playwright` 点击。

:::caution[使用范围]
本文代码仅用于自有系统、测试环境或已获得明确授权的自动化测试。验证码属于访问控制措施，请勿用于绕过第三方网站限制或进行未经授权的数据采集。
:::

## 先看效果

![效果](./img/1.GIF)

## 一、问题建模

识别器接收两张图片：

- `target_path`：只包含目标符号的提示图。
- `background_path`：包含 2×2 四个候选区域的验证码截图。

输出值是匹配候选项在背景截图中的中心坐标 `(x, y)`。如果所有候选项的外观相似度都低于阈值，则返回 `None`，避免低置信度误点。

整个处理链路可以概括为：

- 目标点击图案
  - 灰度化 ──> Otsu 二值化 ──> 连通域去噪 ──> 紧边界裁剪 ──> 64×64 归一化

- 候选图背景
  - 固定阈值分割 ──> 轮廓检测 ──> 定位四个候选框 ──> 逐项执行同样的符号处理

最后计算三路相似度，加权评分。返回候选图中识别选中的中心坐标。

**这里有两个重要前提**：背景图中必须恰好存在 4 个较大的候选块，并且它们能按上下两行排列成 2×2 网格。**如果验证码版式改变，需要相应调整候选区域检测逻辑。**

## 二、依赖与调用方式

安装依赖：

```bash
pip install opencv-python numpy
```

最小调用示例：

```python
recognizer = Geetest4Recognizer(min_appearance=0.70)
center = recognizer.find_matching_center(
    "target.png",
    "background.png",
)

if center is not None:
    print(f"应点击的坐标为: {center}")
```

在 Playwright 中，如果 `background.png` 是对验证码元素本身截图得到的，那么返回值也是相对该元素左上角的坐标，可以直接用于元素内点击：

```python
target_element.screenshot(path=target_path)
background_element.screenshot(path=background_path)

center = recognizer.find_matching_center(target_path, background_path)
if center is not None:
    x, y = center
    background_element.click(position={"x": x, "y": y})
```

这里必须保证“截图元素”和“点击元素”是同一个元素，否则坐标原点不同，还需要额外做偏移换算。

## 三、结果数据结构

每个候选项的计算结果由不可变数据类 `_MatchResult` 保存：

```python
@dataclass(frozen=True)
class _MatchResult:
    index: int
    position: str
    bbox: tuple[int, int, int, int]
    center: tuple[int, int]
    score: float
    appearance_score: float
    hash_score: float
    pixel_score: float
    component_score: float
```

字段含义如下：

| 字段 | 含义 |
| --- | --- |
| `index` | 候选项序号，从 1 开始 |
| `position` | 左上、右上、左下或右下 |
| `bbox` | OpenCV 矩形框 `(x, y, width, height)` |
| `center` | 候选框中心点，也是最终点击坐标 |
| `hash_score` | 感知哈希相似度 |
| `pixel_score` | Dice 像素重合度 |
| `component_score` | 连通域结构相似度 |
| `appearance_score` | 用于可信度过滤的外观分 |
| `score` | 用于最终排序的综合分 |

保存所有中间分数的好处是便于调试。识别失败时，不只是得到一个 `None`，还可以从日志中判断究竟是哈希、像素重合还是结构特征拉低了结果。

## 四、定位四个候选区域

### 1. 固定阈值二值化

背景截图中的候选块通常与浅色背景有明显差异。代码使用阈值 `250` 做反向二值化：

```python
_, block_mask = cv2.threshold(
    background,
    250,
    255,
    cv2.THRESH_BINARY_INV,
)
```

灰度值高于 `250` 的近白色背景变为黑色，其余内容变为白色。这样候选块就成为适合轮廓检测的前景区域。

此处使用固定阈值，而后面提取符号时使用 Otsu 自适应阈值，两者解决的问题不同：这里要利用页面背景颜色稳定的特点找大块区域；后面则要适应不同图标自身的亮度变化。

### 2. 外轮廓与面积过滤

```python
contours, _ = cv2.findContours(
    block_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)
min_area = background.size * 0.05
```

`cv2.RETR_EXTERNAL` 只取最外层轮廓，避免候选块内部的符号产生嵌套轮廓。随后只保留面积不小于整张图 5% 的轮廓，小噪点和文字不会被当作候选框。

代码要求最终恰好得到 4 个矩形，否则立即抛出异常。这种严格检查看起来不够“宽容”，但能阻止版式识别错误后继续产生错误点击。

### 3. 按 2×2 网格排序

OpenCV 返回轮廓的顺序没有业务含义，因此需要主动排序：

1. 以背景图高度中线为界，拆分为上、下两组。
2. 每组按矩形的 `x` 坐标从小到大排序。
3. 拼接成左上、右上、左下、右下。

这样 `_POSITION_NAMES` 和候选项序号才能保持稳定，日志也更容易阅读。

## 五、提取符号前景

候选框中仍包含留白、边框和噪点，不能直接拿原图比较。`_extract_symbol_mask` 完成了三步清洗。

### 1. Otsu 自动分割

```python
_, mask = cv2.threshold(
    image,
    0,
    255,
    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
)
```

Otsu 算法根据当前图片的灰度直方图自动寻找阈值。加上 `THRESH_BINARY_INV` 后，较深的符号成为白色前景，较浅的背景成为黑色。

### 2. 连通域去噪

```python
component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
min_area = max(2, round(image.size * 0.00008))
```

这里采用 8 邻域连通域分析。面积小于阈值的区域被认为是噪点，其余连通域重新画入一张干净掩膜。阈值与当前图片面积成比例，同时至少为 2 个像素，兼顾不同截图尺寸。

需要注意的是，这一步不会只保留最大连通域。目标符号可能天然由多个不相连的笔画组成，如果只保留最大块，结构信息反而会被破坏。

### 3. 紧边界裁剪

通过所有前景像素的最小、最大坐标裁剪掩膜：

```python
ys, xs = np.where(cleaned > 0)
return cleaned[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
```

裁剪后只保留符号本体，消除了原图四周留白不同带来的位置干扰。如果完全没有检测到前景，则抛出 `ValueError`，避免后续出现除零或空数组问题。

## 六、归一化尺寸与位置

目标图和候选图的尺寸通常不完全一致。代码把紧边界掩膜缩放并居中到 `64×64` 画布中：

```python
scale = min((size - 2 * padding) / width, (size - 2 * padding) / height)
```

默认四周预留 8 像素边距，可用区域为 `48×48`。取宽、高两个缩放比例中的较小值，可以在不改变长宽比的前提下让整个符号完整放入画布。缩放完成后再次二值化，减少插值产生的灰度边缘影响。

这一步同时解决了三个变量：

- **尺度差异**：统一到相同画布大小。
- **平移差异**：统一放到画布中心。
- **长宽比保护**：不会把高瘦图标强行拉成方形。

但它不会消除旋转和镜像差异。如果验证码会随机旋转图标，还需要加入角度校正或多角度匹配。

## 七、三路相似度

单一指标很容易在相似图标之间误判，因此代码并行计算三类特征。

### 1. 感知哈希相似度

首先把归一化掩膜缩小到 `32×32`，然后执行二维离散余弦变换 DCT：

```python
frequencies = cv2.dct(resized.astype(np.float32))[:8, :8].reshape(-1)
frequencies = frequencies[1:]
return frequencies > np.median(frequencies)
```

DCT 会把图像从空间域转换到频率域，左上角 `8×8` 区域保存主要的低频结构。第一个值是代表整体亮度的直流分量，信息区分度较低，所以将其移除，最后得到一个 63 位布尔哈希。

两个哈希的相似度就是 1 减去不同位所占比例：

```text
hash_similarity = 1 - HammingDistance / 63
```

pHash 更关注整体轮廓，对少量像素噪声和轻微缩放比较稳定。

### 2. Dice 像素相似度

Dice 系数直接衡量两个二值前景的重叠程度。设目标前景像素集合为 `A`，候选项为 `B`：

```text
Dice(A, B) = 2 × |A ∩ B| / (|A| + |B|)
```

完全重合时得分为 1，没有重合时为 0。它对细节差异敏感，正好补充 pHash 偏重整体结构的特点。

由于 Dice 依赖像素对齐，前面的紧边界裁剪、等比例缩放和居中非常关键。没有归一化时，即使两个符号相同，也可能因为位置偏移得到很低的分数。

### 3. 连通域结构相似度

有些符号整体轮廓相近，但笔画的断开方式不同。代码为原始紧边界掩膜提取三组结构特征：

- 连通域数量。
- 各连通域高度降序排列后，相对于最大高度的比例序列。
- 连通域平均“纵向程度”，即 `log1p(min(height / width, 10))`。

对应的三个子分数为：

```text
count_score       = 1 - |目标数量 - 候选数量| / max(目标数量, 候选数量, 1)
height_score      = 1 - mean(|目标高度签名 - 候选高度签名|)
verticality_score = exp(-|目标纵向程度 - 候选纵向程度|)
```

高度签名长度不一致时，短的一方用 0 补齐。最终结构分采用以下权重：

```text
component_score = 0.55 × count_score
                + 0.30 × height_score
                + 0.15 × verticality_score
```

连通域数量的权重最高，因为它对“一个整体”和“多个分离笔画”的区分最直接；高度签名和纵向程度用于进一步区分局部形态。

## 八、为什么要做两阶段决策

代码没有直接选总分最高的候选项，而是先按外观分过滤，再按综合分排序。

外观分只包含 pHash 和 Dice：

```text
appearance_score = 0.55 × hash_score + 0.45 × pixel_score
```

综合分加入了结构特征：

```text
score = 0.50 × hash_score
      + 0.40 × pixel_score
      + 0.10 × component_score
```

选择过程如下：

1. 只保留 `appearance_score >= min_appearance` 的候选项。
2. 如果没有合格项，输出最接近项的分数并返回 `None`。
3. 如果存在合格项，从中选择综合分 `score` 最高的候选项。

这种设计把“是否足够像”和“合格项中谁更像”分开了。结构特征只占总分 10%，用于接近候选项之间的微调；它不能让一个外观明显不匹配的候选项仅凭连通域结构蒙混过关。

默认阈值是 `0.68`，实际调用时可以提高到 `0.70` 来降低误点率。阈值越高，错误点击越少，但返回 `None` 和刷新验证码的次数会增加。

## 九、坐标为何可以直接点击

候选框由 `cv2.boundingRect` 返回：

```python
(x, y, width, height)
```

中心坐标计算为：

```python
center = (x + width // 2, y + height // 2)
```

只要背景截图直接来自 Playwright 的 `element.screenshot()`，图片左上角就是元素内坐标 `(0, 0)`。因此识别器返回的中心点恰好符合 `locator.click(position=...)` 所需的相对坐标，不需要换算成整个页面的绝对坐标。

中心点击也比点击某个前景像素更稳健：图标笔画可能靠近边缘，而候选区域中心通常有足够大的可点击范围。

