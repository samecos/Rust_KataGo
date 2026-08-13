# KataGo Rust 移植文档

本目录包含 KataGo C++ 引擎向 Rust 移植的完整方案。

## 文件说明

| 文件 | 内容 |
|------|------|
| `rust-port-plan.md` | **主方案**：目标、范围、策略、模块计划、路线图、风险与验收标准。 |
| `module-mapping.md` | **C++ → Rust 文件级映射表**：按模块列出所有源文件的目标 Rust 路径与状态。 |
| `ffi-guide.md` | **FFI 桥接规范**：如何在移植期间让 Rust 临时调用未迁移的 C++ 模块（主要是 GPU 后端）。 |

## 快速开始

1. 先阅读 `rust-port-plan.md` 把握整体策略与阶段划分。
2. 根据当前工作模块，查看 `module-mapping.md` 找到对应源文件。
3. 若涉及 GPU 后端桥接，参考 `ffi-guide.md`。

## 维护约定

- `module-mapping.md` 中的“状态”列应随开发进度每周更新。
- 新增移植相关文档可直接放到本目录，并在此 `README.md` 中登记。
