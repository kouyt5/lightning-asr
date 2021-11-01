# 语言模型生成

文本样式:
```
杉 杉 控 股 认 为 购 额 占 到 百 分 之 六 十
或 是 看 准 了 新 能 源 汽 车 发 展 时 机
却 因 环 评 公 示 期 间 社 会 反 响 大
被 上 海 临 港 地 区 开 发 建 设 管 理 委 员 会 停 止 审 批
每 日 经 济 新 闻 记 者 注 意 到
提 高 油 品 标 准 和 质 量
```
需要用空格隔开

## 训练
参考github仓库: https://github.com/kouyt5/kenlm-docker
2gram
```
docker run -it --rm -v ${PWD}/ckpt/lm:/workspace 511023/kenlm:latest -o 2 --text /workspace/train.txt --arpa /workspace/2.arpa
```