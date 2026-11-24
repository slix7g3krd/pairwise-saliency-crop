# pairwise-saliency-crop

## 领域

游戏、图形与多媒体

## 主题

显著性自动裁剪

## 说明

坐标反映射、用户优先与逐图缓存。

初始状态下 `imaging/saliency_crop.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
