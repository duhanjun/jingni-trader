"""
PIT (Point-in-Time) 数据适配器
==============================

借鉴自 Microsoft Qlib 的 PIT 数据设计：
https://qlib.readthedocs.io/en/latest/advanced/pit.html

背景
----
金融数据存在严重的**未来信息**问题：
- 财务数据：季度报告通常在季末后 30-60 天才发布
- 指数成分：调整生效日与公告日存在时滞
- 宏观数据：发布频率与公布日不一致

naive 的 ``merge`` 会引入 look-ahead bias，导致回测结果虚高。

Qlib 的解决方案
---------------
1. 数据存储时打上"可被使用的时间戳" (announcement time)
2. DataLoader 拉取时按"asof"过滤: ``data[announce_time <= asof_date]``
3. 配合时间错位的算子避免 look-ahead

本模块提供轻量级的 PIT 适配器
----------------------------
- ``PITDataAdapter`` 把普通数据 + 公告日列 转成 PIT 视角
- ``asof`` 接口支持任意时间点的"实盘可见"快照
- 不依赖外部数据库, 单机即可运行
"""

from .pit_adapter import PITDataAdapter, PITField

__all__ = ["PITDataAdapter", "PITField"]
