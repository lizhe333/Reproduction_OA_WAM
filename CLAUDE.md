# OA-WAM 复现项目

## 工作区约定
- 所有共享规格在 specs/ 目录，Agent启动时须先读取
- 子系统接口定义在 specs/04-interface-contracts.md
- 完成后必须：1) 更新 specs/00-project-status.md  2) 写 handoffs/ 交接文档
- 代码放在 src/<subsystem>/，镜像结构

## 当前阶段
阶段一：深度理解

## 关键约束
- 感知栈（SAM3/DINOv3/VQ-GAN/Qwen3-VL）全部冻结，无需训练
- 主干基于7B Chameleon风格，使用LoRA微调
- 训练两阶段：Stage I（VQ-GAN重建）可跳过，直接从Stage II开始
