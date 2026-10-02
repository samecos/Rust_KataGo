# 统一后端的有限候选离线流程

**当前执行状态：用户恢复后的独立速度验证已完成一轮。** 48 个配置初筛、5 组正式 ABBA、2 组独立更长确认；5070 Ti / C32 上 B15/B3、B15/B8 混合候选确认吞吐提高 1.84% / 1.75%，B11 未确认提速。结果与可复用配置身份在 `target/unified-quant-speed-only-final-r1`，详见 [速度验证](RustGo统一后端速度验证.md)。脚本 `scripts/benchmark_quant_speed.py` 与 `scripts/prepare_quant_speed_from_profiles.py` 提供独立有限复测，无精度前置。原失败入口不重启；下方“暂停”与旧 campaign 的在途描述均为历史状态。

**当前范围（2026-10-01 用户调整）：只验证速度，精度全部由用户验证。** 下方精度采集、数值比较、selection/holdout 门和确认流程均为历史路线，不再构成代理性能测量的前置条件。后续复用已有模型、PTQ 参数及候选配置，以相同输入、物理 batch、并发和缓存条件对比现有 FP16/INT8 后端；预热、GPU 同步和交替重复测量后，报告端到端延迟、吞吐、加速比例及波动。Tensor Core profiling 单独用于解释实际路径。精度状态标为「用户验证／本次未评估」，旧失败、锁、预算、消费和截止保留，不重启旧单次入口。当前实验保持暂停，本次仅调整范围；恢复后走独立性能测量入口。当前目标以[主规格开头](RustGo统一推理优化量化后端.md)为准。

**最新：闭gate诊断已完成，仍未识别新增Job成员。** 首次owned实际0/2331ce，16外层/8内层来源稳定，真实outer/helper venv均0、Job0。两即刻列表仅helper63196；父locator耗16ms之后最终累计total2，未知成员不能补成旧失败的身份，未改原计数门或取得消费者资格。r2仅准备≤2秒有界晚观察。新真实history/productionledger/GPU/ABBA0，原失败、采集、消费和截止保持；见规格§8.104。

**最新：完整消费者首次CPU验证在放行前停止，未取得资格。** 一次性计划首次freeze actual0/08e484，六新源AST/compile通过，2962来源首末稳定、绑定精确2949生产库存。session69622 actual1/0d2f03：首方法prepare的Job放行前active2/total2，1error、另三项未跑；gate未尝试写入，第二成员身份未观察，不能猜测或放宽计数门。phase只有cleanup退出3758096385和Job0；outer/helper的实际wait均为1，旧unknown字段保留。2964来源末核稳定，root `ec9ce497…`，不构成消费者通过、精度或性能收益。原三锁/五份数值/历史消费/总钟保持，新production history/ledger/GPU/ABBA0；独立闭gate成员身份诊断准备中。见主规格§8.103。

**最新：Windows held进程链修复三项通过，bootstrap与固定入口已首次构造成功，完整续接资格仍待验证。** held3根actual0/dd2cb1、六个held wait均0；首轮WinError31失败及未知wait保留。新lifecycle-r2/controller-r3使用同一共享函数。bootstrap首次owned freeze根actual0/38d8bc，原2929库存全保留、2947静态来源首末实读稳定，生成entry只替换唯一bootstrap字面Source。generated entry尚未运行；真实ACK/退出释放及隔离历史/collector边界的完整Owner/driver/facade流程检查正在实现。旧失败/五数值/锁/消费/总钟保持，新history/ledger/GPU/ABBA0。详见规格§8.101～8.102。

**最新：Owner／facade／driver完整接线源码已通过联合静态复核，未运行。** 已修首次lease前准入顺序；保留原replay、旧锁、Prepared字段、失败消费和同一截止，C8后补固定独立C3 comparison。新entry/controller、venv实际held身份链与终态validator仍待完成，现有组件检查不替代完整消费者资格。旧三锁再次读验不变、业务claim不存在，新GPU/ABBA0。详见规格§8.99。

**最新：Windows管道三项首次真实CPU检查通过。** actual exit0／`f6d12c`，3 PASS、20来源稳定；144080B双向往返、原截止下真实取消完成995后释放、错误创建时间/明确Job精准拒绝均通过。三direct child实际0、各Job与外层Job active0；独审21小结果/日志稳定。结果`0f396264…`、根退出`71e512ff…`。仅通信与清理资格，Owner/facade/driver业务接线仍在进行，原失败/五数值/消费/总钟不变，业务claim/新GPU/ABBA0。详见规格§8.98。

**最新：外层路由拆分首次4项CPU通过，与前两节累计13项。** actual exit0／`fb6e7a`，20来源稳定；结果`2e4ae00e…`、根退出`2f2c0760…`。原完整active契约、私有mutator及锁内/写后replay不改，仅减少reserve/compare外层各一次传递来源扫描，明确观察时序改变，未实测提速或接入业务。Windows身份认证管道静态完成，真实IPC三项及Owner/driver/facade接线待完成。原首组失败、五份数值及原消费/截止保持，新GPU/ABBA0。详见规格§8.97。

**最新：接续控制核心的首次5项夹具CPU检查通过，与暂缓矩阵4项合计9项。** actual exit0／`835d12`、24来源稳定，结果`16e623e7…`、根退出`852fde1d…`。覆盖固定唯一claim、原replay同边界append、中文事件摘要、失败消费保留和成功收尾的来源/截止检查；使用真实临时小文件及合成admission/replay/append，不代表真实接续执行。业务claim未创建、旧锁未改，跨进程身份认证通道、实际driver/facade及owned组终态仍待完成，未新增GPU或ABBA。详见规格§8.96及ROOT的`current_work`入口。

**最新：全部11行暂缓矩阵阶段的首次4项CPU检查通过，尚未接入实际驱动。** 新阶段保留完整11事件、每行本地来源核验和原finalization，将该阶段完整replay从11次改为前后2次；显式绑定同一截止，阶段context和回执成功退出后才发布outcomes。actual exit0／`f8cebf`、18来源稳定，结果`35efebf9…`、根退出`70e83121…`。检查使用真实固定Prepared/catalog及回调替身，不能当作真实history或运行提速；阶段内传递来源的观察时序与旧逐行全量扫描不同。原首组失败及五份数值结果保留，新consumer／claim／GPU／ABBA未执行。详见规格§8.95。

**最新：6项路径差分检查与4项所有权证据检查首次全部通过，尚未接入业务。** 原完整清单固定A/B/B/A对照返回相同81,661条摘要，路径候选描述性均值优势约1.9%，与约2.0%首尾漂移接近，不采用、不外推整轮提速。原会话工具记录已恢复旧helper→venv→inner的PID/命令链；纯准入在19固定输入上通过，错误身份/非空Job/扩大范围被拒绝。后续采用保留原v4身份、旧三锁和原总钟的单outer successor协议方向，真实claim和driver/facade接线仍在静态实现。旧B11/C3失败与5份数值结果保持，新GPU/ABBA为0。见[主规格 §8.94](RustGo统一推理优化量化后端.md)。

**部分终态独审已完成。** 204项有限一致性检查通过，确认5份已完成结果的预约、采集和数值记录链及Prepared绑定；旧FP16没有发布预约、启动事件或采集目录。42份证据和审计实现前后稳定，审计期间AGENTS更新单列保留。原整组仍为exit1失败，新ABBA0；此次不替代历史全量末核或raw复算。报告位于`target/unified-quant-selection-b11-c3-partial-audit-r1/report.json`，见[主规格 §8.90](RustGo统一推理优化量化后端.md)。

**最新：并行来源核验的8项合成CPU检查首次全部通过。** 外层actual exit0，helper及owned venv均实际exit0，Job终态active0，15份绑定来源稳定。此次验证并发上限、独立快照、失败顺序、原截止及join，尚未验证实际Source文件回调或读取收益，也未接入业务。独立CPU适配器收尾写入后的截止复核缺口已在r2隔离修正，未运行，原8项不重跑。原selection失败及5份数值结果保持，见[主规格 §8.91](RustGo统一推理优化量化后端.md)。

**最新终态：原 B11/C3 流程达到绝对工作截止，以 exit1 停止。** 北京时间2026-10-01 03:48:07，session28076实际exit1；Job内进程已清理至0，正常helper/venv退出未观察，driver整体result缺失。参考、3个混合候选、旧INT8共5份数值结果保留，旧FP16的reserve操作被中断且无结果；事件止81，未见旧FP16预约或采集目录，部分账本独审尚在进行。原入口不得重启或重置截止，下一组success-only freezer不能消费当前失败；新ABBA仍0。来源并行核验原型已静态完成，有限CPU资格待运行，尚未接入。见[主规格 §8.90](RustGo统一推理优化量化后端.md)。下列LIVE条目均为此前历史观察。

**最新续更：旧FP16检查已PASS，reserve操作已开始，尚未确认永久预约。** CPU44 `check`返回PASS，CPU45 `reserve`已有started；第82事件及实际启动尚未观察，不能据操作启动记录宣称预约已消费。独审仅核CPU44 result/started及CPU45 started三小文件，前后稳定、绑定匹配，不证明新OS退出或整组终态。原session28076轮询7a594d仍LIVE，随后时钟记录北京时间2026-10-01 03:44:16；新永久预约仍5、ABBA0，继续原会话、不重启或改消费与截止。见[主规格 §8.89续更](RustGo统一推理优化量化后端.md)。

**最新续更：第三混合比较后检查已PASS，旧FP16检查已启动。** CPU43 `check`返回PASS；旧FP16 `b11-legacy-fp16-c32`／`005-collect`的CPU44 `check`已启动，尚未观察到第82预约事件、实际启动或整体终态。原session28076轮询437ed0仍LIVE，随后时钟记录北京时间2026-10-01 03:36:09。独审仅核CPU43 result/started及CPU44 started三文件，前后稳定、绑定一致，未跟读引用或raw，历史退出不作为新OS退出证明。参考、3个混合候选及旧INT8数值通过记录保留，新预约5、ABBA0保持，继续原会话、不重启或改消费与截止。见[主规格 §8.89续更](RustGo统一推理优化量化后端.md)。

**最新续更：第三混合候选比较末核已PASS，旧FP16仍待完成。** CPU42 `compare`实际返回PASS，driver0020为`NUMERIC_VALIDATED`、gate PASS并绑定第81事件；CPU43 `check`已启动，整组尚未终态。当前参考、3个混合候选及旧INT8均有通过数值记录，旧FP16待完成；数值通过不构成性能或holdout采用依据。原session28076轮询8cc0d5仍LIVE，随后时钟记录北京时间2026-10-01 03:27:28；新预约5、ABBA0保持，继续原会话、不重启或改消费与截止。见[主规格 §8.89续更](RustGo统一推理优化量化后端.md)。

**最新：第三混合候选的selection数值事件已PASS，CPU42末核尚未完成。** `b11-c3-mix-003`（11 INT8/22 FP16 FFN）的第81个`NUMERIC_COMPARED`事件记录32盘2004局面`DIAGNOSTIC_PASSED/PASS`：最大白方胜率差2.1202921867个百分点（门限6），最大目差1.40966796875目，Top1一致1987/2004（99.1516966068%），exact输出0/2004。数值来自事件，独审仅核三事件的SHA、链及身份，未独立复算raw。CPU42末核、driver0020及整组结果尚未见；原session28076轮询595adc仍LIVE，随后时钟记录北京时间2026-10-01 03:19:18。事件PASS不代表操作终态或性能、棋力、holdout、生产通过；旧FP16未完成，新预约5、ABBA0保持。见[主规格 §8.89](RustGo统一推理优化量化后端.md)。

**最新续更：第三混合候选入账末核已PASS，数值比较已启动。** CPU41 `ingest-collection`返回PASS，CPU42 `compare`已启动，数值结果尚未观察。独审仅核CPU41 result/started及CPU42 started三小文件，前后SHA稳定、绑定匹配；历史replay退出不能当成新OS退出，也不证明数值、性能或整组终态。原session28076轮询1c9535仍LIVE，随后时钟记录北京时间2026-10-01 03:08:20；新永久预约仍5、ABBA0，继续原会话、不重启或改消费与截止。见[主规格 §8.88续更](RustGo统一推理优化量化后端.md)。

**最新续更：第三混合候选采集后检查已PASS，第80采集事件已写入。** CPU40返回PASS，v4第80事件为`COLLECTION_OBSERVED`，原始PB及恢复执行身份已验证，`numeric_gate=NOT_COMPARED`。独审仅核第79、第80事件及report三文件，前后SHA稳定，canonical链与身份、report/profile/registration/child-exit Source一致，未跟读引用或raw；不证明CPU41末核、数值、性能或整组终态。CPU41 `ingest-collection`结果和第81数值比较事件仍未见。原session28076轮询0b5b27仍LIVE，随后时钟记录北京时间2026-10-01 02:59:56；新永久预约仍5、ABBA0，继续原会话、不重启或改消费与截止。见[主规格 §8.88续更](RustGo统一推理优化量化后端.md)。

**最新续更：第三混合候选已完成真实采集，尚未通过数值或性能验证。** `b11-c3-mix-003`的native PID46672由supervisor观察exit0，driver0019记录`PROCESS_EXITED`、returncode0；已保存32盘2004局面、668×B3、4009输出文件/41,789,924字节，实际profile匹配catalog。report仍为`COLLECTED_UNVERIFIED`；独立五文件复核仅核链、身份与计数，未复算raw。原session28076最新轮询4e9e11仍LIVE，随后时钟记录北京时间2026-10-01 02:50:33；整组未终态，继续原会话、不重启或重做已消费入口。见[主规格 §8.88续更](RustGo统一推理优化量化后端.md)。

**最新：第三混合候选已启动原生采集，尚未观察退出或采集完成。** `b11-c3-mix-003`的CPU39启动前检查已PASS；native launch记录PID46672，root既有精确PID观察与冻结exe镜像相符，claim及supervisor attempt均为`CONSUMED_NO_RETRY`。启动证据来自launch与该PID观察，driver0018 `ATTEMPT_STARTED`单独不证明Popen或GPU执行，root也未持有native进程句柄或观察其退出。原session28076在北京时间2026-10-01 02:36:20经d54e9b确认仍LIVE，未见native退出、report、stage、driver0019、CPU40及整体结果。新永久预约仍5，参考、两混合候选及旧INT8数值结果保留；第三候选GPU推理完成、数值、性能及holdout未认证，继续原会话、不重启或重做已消费入口。见[主规格 §8.88](RustGo统一推理优化量化后端.md)。

**最新续更：第三混合候选的预约后检查已PASS，启动前检查仍在途。** CPU38 `reservation-head`返回PASS，driver0017确认`RESERVED_BEFORE_EXTERNAL_LAUNCH`，已消费且不可重试、尚未启动执行或允许发布；CPU39 `check`已启动，尚未观察其result、driver0018或实际启动。原session28076在北京时间2026-10-01 02:23:35经493391确认仍LIVE；新增永久预约仍5，参考、两混合候选及旧INT8数值完成证据保留，第三混合、旧FP16、完整矩阵、性能、holdout及生产未完成。继续原会话，不重启或改消费与截止。见[主规格 §8.87续更](RustGo统一推理优化量化后端.md)。

**最新：第三个B11/B3混合候选已永久预约，尚未观察实际采集启动。** `b11-c3-mix-003`／`004-collect`已写入v4第79事件，类型为`ATTEMPT_RESERVED`，新增永久COLLECTION预约累计5；CPU37 `reserve`已PASS，CPU38 `reservation-head`已启动、尚无终态。11 INT8/22 FP16 FFN是候选B3的静态保存路由，不能当成本次运行trace或性能证据。原session28076在北京时间2026-10-01 02:14:49经471342确认仍LIVE；参考、两混合候选及旧INT8数值完成证据保留，第三混合、旧FP16、整组、其它负载、性能及holdout仍未完成。29项未来CPU、v5、ABBA及profiler未执行，继续原会话、不重启或重置预算与截止。见[主规格 §8.87](RustGo统一推理优化量化后端.md)。

**最新续更：旧INT8 q64比较末核已PASS，下一候选仍仅在冻结计划中。** CPU34 `compare`完整返回PASS，driver0016记录`NUMERIC_VALIDATED`、gate PASS并绑定v4第78事件；CPU35 `check`已启动，尚无整体结果。原session28076在北京时间2026-10-01 01:49:48经19d467确认仍LIVE。当前已完成参考、两混合候选及旧INT8的数值阶段；下一项为`004-collect`／`b11-c3-mix-003`（11 INT8/22 FP16 FFN仅为静态路由观察），尚未观察其预约或启动。新增永久预约仍4，第三混合、旧FP16、全组、其余负载、性能及holdout仍未完成；继续原会话、不重启或改消费与截止。见[主规格 §8.86续更](RustGo统一推理优化量化后端.md)。

**最新：旧INT8 q64控制的selection数值门已通过，CPU34末核仍在途。** v4第78事件为`NUMERIC_COMPARED`，旧控制相对共享统一FP16参考的32盘2004局面诊断为`DIAGNOSTIC_PASSED/PASS`：最大白方胜率差2.85074413个百分点（门限6），Top1一致1976/2004（98.60279441%），exact输出0/2004、非逐位相同。原session28076在北京时间2026-10-01 01:42:14经0f0bbd确认仍LIVE，CPU34 result、driver0016及整体总结果仍未见。此结果仅为selection诊断，FP32未评估、性能及棋力未测、holdout及生产未通过；新永久预约仍4，29项未来CPU、v5、ABBA及profiler未执行，继续原会话、不重启。见[主规格 §8.86](RustGo统一推理优化量化后端.md)。

**最新续更：旧INT8 q64控制入账末核已PASS，数值比较已启动。** CPU33 `ingest-collection`返回PASS，CPU34 `compare`已有started，尚无result、第78事件或整体终态；第77事件的`numeric_gate=NOT_COMPARED`保持原记录。原session28076在北京时间2026-10-01 01:32:17经74cbaa确认仍LIVE。CPU33结果不是独立OS退出证明，旧replay的退出记录不能代替；新永久预约仍4，旧消耗及参考、两候选数值PASS均保留，整体、性能和holdout未通过。继续原会话，不重启或改消费与截止。见[主规格 §8.85续更](RustGo统一推理优化量化后端.md)。

**最新续更：旧INT8 q64控制的采集已写入v4第77事件，CPU33仍未终态。** 该事件为`COLLECTION_OBSERVED`，原始PB及执行身份已验证，`numeric_gate=NOT_COMPARED`；事件观察中的`process_start_after_reservation`仍为`NOT_PROVEN_BY_COLLECTION_REPORT`，不能套用restored的`actual_child_exit`证明。原session28076在北京时间2026-10-01 01:25:58经1d74da确认仍LIVE，CPU33尚无result，无数值、整组或性能终态。新永久预约仍4，29项未来CPU、v5、ABBA、profiler及holdout未执行，继续原会话、不重启或改消费与截止。见[主规格 §8.85续更](RustGo统一推理优化量化后端.md)。

**最新续更：旧INT8 q64控制已完成真实采集，入账仍在途。** `b11-legacy-int8-q64`／`003-collect`的driver0015记录`PROCESS_EXITED`、returncode0；32盘2004/2004请求完成，最终heartbeat记录2004行、252个batch、失败0、在途0，`drained=true`，report仍为`COLLECTED_UNVERIFIED`。CPU31及CPU32检查已PASS，CPU33 `ingest-collection`仅有started、尚无result；未观察CPU34启动。原session28076在北京时间2026-10-01 01:19:48经848b0d确认仍LIVE，v4第77/78事件、driver0016及总结果均不存在。参考及两混合数值PASS保留；此旧控制尚未数值比较或性能认证，整组、holdout及生产未完成，29项未来CPU、v5、ABBA及holdout未执行。继续原会话，不重启或改消费与截止。见[主规格 §8.85续更](RustGo统一推理优化量化后端.md)。

**最新续更：旧INT8 q64控制的预约后检查已PASS，启动前检查仍在途。** CPU30 `reservation-head`已返回PASS，driver事件0013为`ATTEMPT_RESERVED`，其receipt明确为`RESERVED_BEFORE_EXTERNAL_LAUNCH`、已消费且不可重试、尚未启动执行、未获发布资格，并与v4第76事件绑定。CPU31 `check`已启动，尚无result或driver0014，未观察实际采集启动。原session28076在北京时间2026-10-01 01:01:06经481db4确认仍LIVE；新增永久预约仍4，继续原会话、不重启或重置消费与截止。参考及两混合数值完成证据保留，整组、完整矩阵、性能、holdout及生产仍未完成。见[主规格 §8.85续更](RustGo统一推理优化量化后端.md)。

**最新：旧INT8 q64控制已完成永久预约，尚未观察实际采集启动。** `b11-legacy-int8-q64`／`003-collect`已写入v4第76个`ATTEMPT_RESERVED`事件，新增永久COLLECTION预约累计4；CPU29 `reserve`结果PASS，CPU30 `reservation-head`已启动、尚无终态。静态配置为`cudaint8backend`、batch8/nnMaxBatchSize8、capacity32/window32，实际batch分布尚未观察。原session28076在北京时间2026-10-01 00:52:25经c18b2d确认仍LIVE，继续同一会话、不重启或重置消费与截止。FP16参考及两混合候选数值完成证据保持，整组、完整性能、holdout及生产未完成；29项首次CPU、v5、ABBA及profiler均未执行。见[主规格 §8.85](RustGo统一推理优化量化后端.md)。

**最新续更：第二个B11/B3混合候选的比较末核已PASS，后续检查已启动。** `b11-c3-mix-001`的CPU26 `compare`返回PASS，driver事件0012为`NUMERIC_VALIDATED`、gate PASS，并与v4第75事件绑定；CPU27 `check`已启动。原session28076在北京时间2026-10-01 00:28:19经cb1872确认仍LIVE，整组尚未终态。当前已完成FP16参考自验及两混合候选数值阶段，剩余旧控制、第三混合、完整性能、holdout及生产仍未完成；继续原会话、不重启。独立复核仅覆盖四份小回执，CPU result不等于独立OS退出证明；29项首次CPU、v5、ABBA及profiler均未执行。见[主规格 §8.84续更](RustGo统一推理优化量化后端.md)。

**最新：第二个B11/B3混合候选的数值门已通过，CPU26末核尚未完成。** `b11-c3-mix-001`（3 INT8/30 FP16 FFN）的第75个`NUMERIC_COMPARED`事件已生成，32盘2004局面的实际metrics gate为`PASS/DIAGNOSTIC_PASSED`：最大白方胜率差0.87857246个百分点，Top1一致1998/2004，exact输出0/2004，非逐位相同。CPU26 `compare`结果及driver/controller总结果仍不存在；原session28076在北京时间2026-10-01 00:20:35经cfb28b确认仍LIVE，继续同一会话、不重启。性能为`NOT_MEASURED`、FP32为`NOT_EVALUATED`，不能据误差较小选择性能胜者；holdout及生产未通过，29项首次CPU、v5、ABBA及profiler均未执行。见[主规格 §8.84](RustGo统一推理优化量化后端.md)。

**最新续更：第二个B11/B3混合候选入账末核已PASS，数值比较已开始。** `b11-c3-mix-001`的CPU25 `ingest-collection`已返回PASS，CPU26 `compare`已启动、尚无结果；v4仍为74事件，`numeric_gate=NOT_COMPARED`，不能将入账通过写成候选数值通过。原session28076在北京时间2026-10-01 00:11:49经chunk0bb979确认仍LIVE，driver和controller均无总结果，ROOT当前阶段已同步为数值比较在途。继续同一会话、不重启；首mix-002数值PASS保持，完整性能、holdout及生产未通过，29项首次CPU、v5、ABBA及profiler均未执行。见[主规格 §8.83续更](RustGo统一推理优化量化后端.md)。

**最新：第二个B11/B3混合候选已完成原生采集和CPU24检查，第74事件已生成，入账操作仍在途。** `b11-c3-mix-001`（3 INT8/30 FP16 FFN）的native PID8544由supervisor观察exit0，launcher driver0011 returncode0；已保存32盘、668×3=2004局面、4009输出文件，report为`COLLECTED_UNVERIFIED`。CPU24检查已PASS；v4第74个`COLLECTION_OBSERVED`事件已通过root有限身份核对，`numeric_gate=NOT_COMPARED`，CPU25 `ingest-collection`尚无result、CPU26未启动，不能宣称数值通过。原session28076在北京时间2026-10-01 00:03:54经chunk307021确认仍LIVE，外层和driver均无总结果；新永久预约3，继续同一会话、不重启。首mix-002数值PASS保持；完整性能、holdout及生产未通过，29项首次CPU、v5、ABBA及profiler均未执行。见[主规格 §8.83](RustGo统一推理优化量化后端.md)。

**最新：第二个B11/B3混合候选已启动原生采集。** `b11-c3-mix-001`的启动前检查已PASS，native launch记录PID8544，claim为CONSUMED_NO_RETRY；尚未观察到退出、采集报告或数值比较结果。原session28076仍live，继续同一会话，不重启。首mix-002已完成的数值PASS保持，完整性能对照、确认及holdout仍待完成。见[主规格 §8.82](RustGo统一推理优化量化后端.md)。

**最新：第二个B11/B3混合候选已完成永久预约。** `b11-c3-mix-001`采用3个FFN INT8／30个FFN FP16，预约CPU步骤已PASS，正在核对预约后的账本状态；尚未观察到原生采集启动。新采集预约累计3，已完成的是FP16参考与首个mix-002的数值阶段。继续原session28076，不重启或重复已消费项。见[主规格 §8.81](RustGo统一推理优化量化后端.md)。

**最新：首个B11/B3混合候选已完成selection数值验证，实验精度门PASS。** `b11-c3-mix-002`（6个FFN INT8／27个FFN FP16）相对FP16参考，在32盘2004局面上的最大胜率差为1.368234个百分点、平均0.040550个百分点，Top1一致率99.3014%；门限为最大胜率差不超过6个百分点。比较CPU末核现已PASS，driver已写NUMERIC_VALIDATED；原流程进入后续检查，整组仍未终态。该结果不是逐位等价或性能通过。继续原session28076，不重启；完整控制矩阵、独立确认及holdout仍待完成。见[主规格 §8.80](RustGo统一推理优化量化后端.md)。

**最新：首个B11/B3混合候选已完成采集和入账，数值比较已启动。** `b11-c3-mix-002`的native与launcher均实际exit0，668个三行batch覆盖32盘2004局面。v4事件71验证原始输出及恢复执行身份，CPU17入账末核现已PASS；CPU18与FP16参考的比较已启动，尚无裁决或性能结论。原session28076仍live，只继续原会话，不重启。29项首次CPU、v5性能续接、ABBA及profiler仍未执行。见[主规格 §8.79](RustGo统一推理优化量化后端.md)。

**最新：首个B11/B3混合候选已记录原生采集启动。** 同session28076仍live；`b11-c3-mix-002`的启动记录给出native PID81736，但尚无退出或比较完成观察。该候选保存的路由契约为6个FFN INT8、27个FFN FP16，FP16上投影保留融合DualFFN；路由目录不是实际kernel计数器，不据此推断精度或速度。继续原会话，不重启；29项首次CPU及profiler仍未执行。见[主规格 §8.78](RustGo统一推理优化量化后端.md)。

**最新：首个B11/B3混合候选已完成永久预约，原会话继续。** `b11-c3-mix-002`的reserve CPU已PASS，当前在reservation-head步骤；新预约累计2，不是已完成两份采集。首份参考自比较已通过，但混合候选仍无精度结论。完整contract在双seal前精确检查128MiB的修复及消费接线已静态封存独审；首次CPU现为4／15／4／6共29项，尚未执行。后续库和caller分别只用新的cpu-r3／cpu-r2准备器，原未执行版本保留。详见[主规格 §8.77](RustGo统一推理优化量化后端.md)。

**最新：首份B11/B3参考采集的入账和自比较均已完成。** 同session28076仍live，32盘2004局面全部输出一致，原流程已写NUMERIC_COMPARED／NUMERIC_VALIDATED；这是同一参考attempt的自比较，不代表混合候选或整组精度通过。后续CPU检查正在进行，不重启。最终性能计划组装器已静态封存独审，未实际装配或执行；完整contract发布前128MiB检查的最小顺序修复正在独立静态准备，首次CPU资格需按最终字节绑定。见[主规格 §8.76](RustGo统一推理优化量化后端.md)。

**最新：首份参考结果已写入COLLECTION_OBSERVED账本事件，尚未比较。** 原session28076仍live，ingest操作正在末核；事件绑定原32盘2004局面的报告和恢复执行身份，不能据此称精度／性能PASS或重启。最终performance计划组装器正在静态准备，真实四组终态及4／12／4／6项CPU资格仍是后续运行前置。见[主规格 §8.75](RustGo统一推理优化量化后端.md)。

**最新：首份 B11/B3 参考的原生采集和启动器均 exit0，报告覆盖32盘2004局面。** 原session28076仍live，报告状态COLLECTED_UNVERIFIED，校验入账和整组尚未终态，不重启。性能入口、调用器及4／12／4／6项首次CPU准备器均已静态封存独审，新增检查尚未执行。完整四组数值选型、ABBA、独立确认及holdout仍未通过。见[主规格 §8.74](RustGo统一推理优化量化后端.md)。

**最新：首组已为第一份数值采集完成永久预约。** 原session28076仍live；目录登记和预约CPU操作已PASS，B11/B3参考 `b11-c3-mix-000` 的COLLECTION预约已消费，尚无采集完成或整组终态。继续同一会话，禁止重启。新性能入口静态核出1800秒历史上限与长phase冲突，正在独立收紧parent子截止；外层总时限保持。完整matrix flow的两项新检查仅定义，将与caller检查一起首次运行，详见[主规格 §8.73](RustGo统一推理优化量化后端.md)。

**最新：首组 run 历史核验已退出0，原 driver 已进入 RUN_STARTED，整体仍在运行。** 只续看 session28076，不重启；尚无本组采集完成回执。v5性能续接库已静态封存，保留完整44对矩阵并按原数值资格阻止不合格组合预约；新的矩阵driver直接复用原基准与读取器，10＋4项新检查均仅定义。后续两阶段受控入口仍在静态接线，未登记或执行ABBA。见[主规格 §8.72](RustGo统一推理优化量化后端.md)。

**最新：首组 prepare 实际通过，已进入原 run-execution。** 同session28076的prepare helper与owned venv均exit0，1120.469秒、Job最终Active0；实际生成6 COLLECTION／0 ABBA的Prepared，11对性能仍deferred。run阶段已启动但尚无数值或整组终态，继续原session，不重启。见[主规格 §8.71](RustGo统一推理优化量化后端.md)。

**当前：首组 prepare 的两层历史重放已正常退出，原 session28076 仍未终态。** v3／v2 owned子进程均exit0，继续原入口完成准备，不重启。后续三组freezer和性能来源投影均已静态封存独审；新增四项Source检查仍未执行，尚无实际bootstrap或ABBA资格。见[主规格 §8.71](RustGo统一推理优化量化后端.md)。

**最新：首组 B11/B3 完整数值选型已首次启动，尚无终态。** 首组计划实际冻结成功，15,344份输入来源稳定；唯一session28076／producer23464，只续看原session，不重启。计划保留六份数值采集、完整11对性能矩阵，ABBA仍为0；后续按实际终态衔接。四组运营上限8／6／8／6小时，共用28小时总单调时钟，包含组间间隔，不能作为预计耗时。窗口内仅静态并行，详见[主规格 §8.70](RustGo统一推理优化量化后端.md)。

**最新：控制器资格消费者四项 CPU 首次实际通过。** direct exit0（b458e8），2,748份来源稳定，四项无失败／跳过／禁止操作。最小r3保留原Job、gate、超时和执行主体，按来源消费既有局部证据，旧聚合失败保持原样。首组计划正在绑定原registration锁定的driver目录和完整四组时限；尚未启动selection，20次采集／0 ABBA未消费。见[主规格 §8.69](RustGo统一推理优化量化后端.md)。

**最新：单项成员身份诊断已实际通过，确认本次额外成员为三个 conhost.exe。** direct exit0（2f3553），2,786 份来源稳定；七个成员的精确身份和两个 Job 的成员关系均核验，正常等待均 exit0，两个 Job 最终为空。EOF 结果继承且未重跑；此前未记录身份的 PID 仍为未知，不据新结果补写旧记录。正在以这些具体证据修正生产控制器的资格消费，尚未放行 selection GPU，见[主规格 §8.68](RustGo统一推理优化量化后端.md)。

**最新：直接成员诊断已实际推进到完整列表，额外三个活跃进程的身份仍待查明。** 新入口 direct exit1（52b6f3），2,751 份来源稳定。EOF 预期拒绝和清理通过；breakaway 的四个声明角色均实际属于两个 Job，但两份完整列表各有七个活跃成员，不能再把差额解释成 TotalProcesses 重复计数。叶进程已通过精确句柄回收、两个 Job 最终为空。旧入口保留不重跑；后续仅一次新的 breakaway 身份观察，继承 EOF 结果。尚无新 selection GPU 或 ABBA，见[主规格 §8.67](RustGo统一推理优化量化后端.md)。

**最新：v4 账本发布已实际成功，进程树控制首次 CPU 检查失败并保留。** session71940 实际 exit0；新登记继承旧 10 次采集、16 组 ABBA，剩余 20 次采集／0 ABBA，尚未新增预约或 GPU。session39597 的 5 项纯检查和前 6 项真实检查局部通过，最后 EOF 计数断言失败；另发现 breakaway 叶进程缺直接 Job 成员证明。正在独立两 case 中补充进程列表、精确句柄与出生身份观察，不能将 TotalProcesses 当唯一 OS 进程数、提高旧 23 上限或重启旧入口。见[主规格 §8.66](RustGo统一推理优化量化后端.md)。下列在途描述保留历史阶段含义。

**最新：六个选型目录实际全部生成成功，driver 接线十项 CPU 检查通过。** 原根 legacy 路径检查实际通过后，同一 session45554 实际 exit0，19,447 份绑定来源稳定，完整 B11/B15 × B3/B8 数值与性能矩阵保留；三次旧失败不重启。driver r2 十项首次 CPU 检查实际 exit0，覆盖命令投影、真实配置加载及快照依赖来源。v4 控制账本发布首次运行于 session71940/PID7892，尚无终态，只观察同一 session；仍无新 selection GPU 或 ABBA。GPU driver 全树超时终止与实际清理验证尚未完成。见[主规格 §8.65](RustGo统一推理优化量化后端.md)。

**最新：预算修复两项检查已通过，实际目录发现旧配置的相对路径基底错误。** session20922 已实际 exit1；snapshot 中的 legacy 模块将 B11 原配置的相对计划路径解析到了 snapshot 内，原配置和计划内容均未改变。正在显式接回原根模块来源，并核对 catalog／v4／driver 子进程的导入一致性。三次失败原件保留且不重启；目前 0 新目录发布、0 selection GPU、0 新账本操作。见[主规格 §8.64](RustGo统一推理优化量化后端.md)。

**最新：环境修复已实际通过，目录生成转入预算字段兼容修复。** r3新两项CPU检查通过；session32297真实首例越过环境和导入检查，原catalog随后拒绝控制模板中的11组比较（旧模板上限8）。应分别使用B11(4,5)／B15(5,8)合法控制模板预算，以及新候选原(6,11)完整矩阵；r4源派生中，不改旧校验器，不删比较项，不重启失败目录。16份采集登记继续有效；实际六目录成功、v4续接、selection GPU和ABBA尚未完成。见[主规格 §8.63](RustGo统一推理优化量化后端.md)。

**最新：六步CPU目录流程在首例进入catalog前停止，正在修复环境比较。** 运行器4项检查已实际通过；真实首例因Windows环境键名大小写被误拒，原session54368已exit1且保留，0目录发布／Protocol／GPU／账本操作。一次相同venv环境观察确认75键值完全一致，只存在19个键名大小写差异；新r3保留原环境并修正比较和首错记录，不重启旧control。v4入口三个首次CPU检查也已通过，但真实续接尚未执行。详见[主规格 §8.62](RustGo统一推理优化量化后端.md)。

**当前：16份采集登记已实际生成，尚未预约或运行。** 路径修复七项CPU检查通过；freeze-only同一session37895实际exit0，19106来源稳定，结果SHA `3409e2cb550d06bdfb8fd65f3392ab4203b7289e1f41840307748ef6ed9ae2aa`。旧Prepared、16原生预检及执行包验证均复用，失败原件保留。后续六步CPU目录运行器正在修正Windows虚拟环境的进程身份校验，随后才进入catalog／原根账本续接；没有新增selection GPU、ABBA或性能结论。见[主规格 §8.61](RustGo统一推理优化量化后端.md)。以下记录保留历史阶段含义。

**当前：16项 selection CPU 预检已全部通过，采集登记尚未生成。** session20024实际exit0；之后的CPU freeze被Windows路径等价检查拒绝，session94549实际exit1，失败原件保留且未创建GPU/账本预约。正在修复Python读取器对原生extended DOS路径的比较，既有16项预检、B11/B15执行包和独立审计不重跑、不改写。最新细节见[主规格 §8.60](RustGo统一推理优化量化后端.md)。

最新进度：**B11、B15 各八份执行包均完成保存／恢复一致性验证和独立审计**。B15 同一 session35450 实际 exit0，独审5972份来源稳定、440对原始输出及8对包一致。新增36项CPU检查首轮全部通过；16份 selection 配置的 prepare＋CPU preflight 已首次启动 session20024/PID68584，终态待观察，勿重启。当前仍未启动新的 selection GPU 采集或 ABBA；以上不代表模型精度、性能或棋力认证。见[主规格 §8.59](RustGo统一推理优化量化后端.md)。下列在途表述保留其历史阶段含义。

最新进度为 **B15 4/8组完成**：容量3的参考及三个混合候选保存／跨进程恢复均逐位一致，正在容量8参考组（同一session35450，勿重启）。完整逐case环境selection assembler、17child CPU runner、B15独立审计器以及36项首次CPU检查入口已完成静态交付；尚未执行这些新增CPU检查或独立审计。窗口结束后先取得真实GPU总退出，再按实际回执推进，不能据源码交付称选型或性能通过。见[主规格 §8.58](RustGo统一推理优化量化后端.md)。

同一B15新会话已完整通过 **1/8组**（session35450/chunk23450a）：首组容量3参考方案的四阶段actual exit0，保存/跨进程恢复输出与包逐位一致；现运行第二组 `b15-c3-mix-001`，没有总退出。结果仅证明本组执行包一致性，尚非独立精度或性能认证。继续观察原会话，勿重启。

**最新：B15八组GPU保存／恢复验证已首次启动。** CPU派生r1的新目录strict解析误拒原件保留；隔离r2经3项针对性路径检查后实际exit0（session50660/chunk8cd8e9），八组登记已生成。新启动计划SHA `5530d8a8be88330806d0ba2bef9e64abca1409df4fa162a786aa56f98039c267`，session35450/controller PID3196，首组supervisor57152，尚无总终态。只继续观察同一入口，不重启；旧B11成功结果和B15失败消费均保留。窗口内不运行测试或构建，尚无新增精度/性能认证。详见[主规格 §8.57](RustGo统一推理优化量化后端.md)。以下在途或未执行表述均保留其历史阶段含义。

新B15接续派生器与八组运行入口的14项新增CPU检查已首次全部通过，actual exit0/chunk6e215a，2546来源稳定。它们保留B11成功结果和B15失败消耗；完整derive、新GPU登记及运行计划尚未执行/生成。不要直接运行静态入口或重启旧失败尝试。后续逐case环境适配仍为静态增量，见[主规格 §8.56](RustGo统一推理优化量化后端.md)。

**最新：B15修正后的八组真实CPU前置验证已全部通过。** `target/unified-quant-b15-compatibility-cpu-r1` 的一次prepare和八组原exe preflight均actual exit0，session91222最终exit0/chunk351452，3277来源稳定；CPU结果SHA `a9c2bfc35848d09533468b94819836ad8eca913f15358934f1496b5e6122a57d`。新环境只把不适用的DualFFN置0，B11八份成功结果继续复用。尚未执行新的B15 GPU接续、selection或性能验证；原失败及消费必须继承。详见[主规格 §8.55](RustGo统一推理优化量化后端.md)。以下阶段记录保留其历史含义。

本轮终态现已完成独立审计：**8份B11成功、1份B15已消费失败、7份未启动**。16项有限CPU检查全部通过；partial审计核对440对原始输出、88对解码结果、8对执行包一致，4898来源稳定，actual exit0/chunk550b42，报告SHA `0792400f9883c0e23c4bb2cd452d097c920cfcd4d8fde78a4866f35bc81e91a4`。B11原成功证据保留使用；新的B15 DUAL0 prepare已静态实现并通过纯CPU辅助检查，但尚未运行prepare/8个真实CPU preflight或新增GPU尝试。完整selection、性能与holdout仍未通过，详见[主规格 §8.54](RustGo统一推理优化量化后端.md)。

**最新已终态：session69973 actual exit1（chunk6fec6b），8份B11成功、首份B15失败、后7份未启动。** B15配方K512不适用DualFFN固定K384/H1152路径，但旧组装器将两模型统一开启DUALFFN；record在模型上传入口被拒绝。失败已消费model_load/model_upload预约各1，不能当作零消耗重启。原8份B11结果保留；下一步独审本次成功与失败，再用同exe已有的DUAL0合法路径派生新的B15登记，不能直接执行依赖原16组全成功的静态准备工具。真实外层退出回执SHA `aa53534db791b5a833552e0f3a693ec85e76b1f25aee5bac74ede240ea0f2dcf`；总result SHA `69a5a18111de1483fcd2dd9f54ac18f2f7250d7c3f09611464f4ae77399e6715`。以下在途描述保留为历史。

最新在途观察已到 **8/16完整通过**：同一session69973/PID40860（chunk cc1fbd），B11 capacity3和8的参考及各三候选保存/恢复全部逐位一致；当前第9组b15-c3-mix-000，总退出仍未观察。八组实际终态元数据及来源已绑定于 `target/unified-quant-candidate-campaign-launcher-r2/run/progress-08-observation.json`（SHA6341af4783fdc1ecb2b405f8695a8d2d15353a5fcc0da7c3143ca777e3f4bbed），独立原始输出审计待总终态后执行。交接见同目录 `ROOT-LIVE-STATE.json`，禁止重启或重复已消费组。

最新实际运行见[主规格 §8.52](RustGo统一推理优化量化后端.md)：显式环境38项核验及13项CPU检查通过，Windows路径比较窄修后的10项相关检查也通过。首例合法控制续接保留原失败；16组GPU记录/恢复验证已首次启动，session69973/controller PID40860。首例B11 FP16参考capacity3四阶段实际exit0、记录/恢复逐位一致，现已完成1/16并进入第2组；串行总终态待观察。不得重启该入口或重复原attempt；只能继续观察同一会话。下文“尚未运行”描述保留其历史阶段含义，不能覆盖本段在途状态；selection与性能验收仍未完成。

截至 2026-09-27，旧 C1 选型及其 q64 续接已经终态，完整控制矩阵没有可采用的新候选，原尝试不得重跑。后续 12 份逐层混合配方已完成 2,044 个 B1 校准局面的比较，均通过相对同模型 FP16 的最大胜率偏差 ≤6 个百分点门；其 B3/B8 整网数值、最终执行方案的端到端性能、独立确认及留出验收尚未完成。B3/B8 是候选提名时的物理 batch，不能从 B1 的数值结果推断。

同一正常后端已在隔离二进制上通过 B11 FP16、B15 首 FFN INT8、既有 ONNX FP16 的有限跨进程保存／恢复验证：capacity3、B1/B2/B3 双槽，五头原始输出、解码结果和重新导出包逐位一致；独立 CPU 审计 852 份来源前后稳定。它证明有限运行与恢复一致性，不是性能或部署认证。后续 DualFFN 探针记账、固定包 selection 采集接口和 Worker 输出适配已在隔离源码中通过 421 项 Rust CPU 与 32 项 Python 检查、四次编译链接，见[主规格 §8.49](RustGo统一推理优化量化后端.md)。

再续接已完成两模型各 2,004 行 selection 的 CPU 输入导出及独立审计，18,688 份绑定文件前后稳定；只证明 B1 存储输入，不是 B3/B8 推理。新增完整 catalog、历史 continuation 和外层 collection 启动器已隔离实现，177 个唯一 Python 基础单测与 14 项 Rust 父证明检查通过；真实超时清理修复另过 9 项相关检查和一次 timeout-only CPU 检查。历史临时目录隔离的 20 项 mock 及唯一只读重放也已实际 exit0，旧 10/16 消耗和全部负结果保持不变；长流程调用方剩余 5 项 CPU 补检及来源末核现已通过，原检查环境失败与超时记录保留。最终 CUDA CLI 已 Cargo exit0，独立读回来源通过，exe SHA `dd665d03fe31739712a205eaa6f63852dbc27fee339dfe91f38067237e57bfdc`；最初外层 provenance 集合误判的失败保留。详见[主规格 §8.50](RustGo统一推理优化量化后端.md)。同一新 exe 的 16 份真实 CPU preflight 全通过，10 项构造器检查通过，16 份运行计划已生成但尚未 seal。GPU 执行包录制/恢复、真实 catalog／continuation 登记、B3/B8 selection 和完整性能矩阵尚未运行；当前根脚本仍按下述旧种子使用，不能直接启动这 12 份混合候选。

入口为 `scripts/run_quantization_search.py`。它串联连续请求采集、相对同模型 FP16 的数值比较、账本登记、ABBA 和汇总。当前仅执行预先冻结的 v1 种子：全 FP16、全 FFN INT8、hidden ≥384 的 FFN INT8；相同配方会去重，因此当前 B11 为 2 个、B15 为 3 个。

最新同exe RPC适配已在隔离快照完成，19项准入与34项运行配置/报告reader CPU检查通过，见[主规格 §8.51](RustGo统一推理优化量化后端.md)。首次34项的测试语法导入失败保留，修复后才完成实际方法执行。随后第一份GPU计划 `b11-c3-mix-000` 在supervisor环境检查处实际exit1，尚未seal或启动任何native/GPU工作；该attempt永久保留、不得重启，余15份未启动。后续应复用显式 `Popen(env=values)` 入口，并为首例形成有失败来源和零GPU证据的一对一控制续接。当前没有新GPU执行包、selection精度或性能通过，根脚本仍不能直接启动这些混合候选。

这是有限实验流程。下文保留 v1 小语料示例；后文 catalog-v2 入口已接入登记的旧 FP16/INT8 控制，要求完整比较矩阵，不宣称穷尽全部历史配置或找到全局最优。不自动确认胜出者，不读取 holdout 输出，也不生成部署认证。逐层敏感度搜索与最终留出验收仍待完成。MXFP8 可通过统一执行器和独立采集工具实验，尚未进入本工具的候选集。

## 本机小语料流程示例

在 `D:/code/Rust_KataGo` 的 PowerShell 中运行。以下使用现有两盘棋语料的 calibration 部分，只用于验证工具接线；最大物理 batch、请求并发和容量均为 1，每臂仅 32 个计时请求，不能据此采纳性能方案。所有输出目录必须是新的。

```powershell
$python = 'D:/Go/Server/worker/.venv-windows/Scripts/python.exe'
$exe = 'D:/code/Rust_KataGo/target/unified-quant-g3/frozen-r9/katago-rs.exe'
$model = 'D:/code/Rust_KataGo/models/b15-ffn-pruned-a8.bin.gz'
$corpus = 'D:/code/Rust_KataGo/target/ptq-corpus-two-games-r2/manifest.json'
$sgfs = 'D:/Go/FrontEnd/src/fixtures/sgf'
$run = 'D:/code/Rust_KataGo/target/quant-search-example-b15'
if (Test-Path -LiteralPath $run) { throw 'Use a NEW experiment directory' }
New-Item -ItemType Directory -Path $run -ErrorAction Stop | Out-Null

& $exe quant-inspect --model $model --output "$run/model"
if ($LASTEXITCODE -ne 0) { throw 'CPU model inspection failed' }
& $python scripts/plan_quantization_search.py --model $model --model-manifest "$run/model/model-manifest.json" --fp16-template "$run/model/fp16-recipe.json" --corpus-manifest $corpus --sgf-root $sgfs --output "$run/search"
if ($LASTEXITCODE -ne 0) { throw 'CPU planning failed' }

@{
    schema = 'rustgo-quantization-ledger-workload-v1'
    split = 'calibration'; metric = 'p95_latency'
    concurrency = 1; capacity = 1; warmup = 1; cycles = 1
    task_timeout = 180
    max_numeric_evaluations = 3; max_performance_evaluations = 2
    minimum_improvement = 0.01; maximum_relative_spread = 0.05
    smoke = $true
} | ConvertTo-Json | Set-Content "$run/workload.json" -Encoding ascii

$preparedText = & $python scripts/run_quantization_search.py prepare --search-plan "$run/search/search-plan.json" --workload "$run/workload.json" --binary $exe --batch 1 --sgf-root $sgfs --output "$run/prepared"
if ($LASTEXITCODE -ne 0) { throw 'CPU preparation failed' }
$prepared = ($preparedText -join [Environment]::NewLine) | ConvertFrom-Json
$preparedText | Set-Content "$run/prepare-result.json" -Encoding utf8
# Above: CPU only. Below: starts isolated local GPU Workers and the fixed experiments.
& $python scripts/run_quantization_search.py run --prepared "$run/prepared/prepared.json" --expect-manifest-sha256 $prepared.prepared.sha256
if ($LASTEXITCODE -ne 0) { throw 'Experiment stopped; retain the failure artifacts' }
```

`prepare` 不启动子进程或 GPU：重新核验完整搜索计划、原模型/SGF/语料和候选，冻结二进制、Python、工具源码、环境、工作负载及完整命令。`run` 再次验证这些条件。升级源码后，旧准备文件会拒绝运行；须从原始输入生成新的计划和准备产物，不能改旧 SHA 来绕过检查。

每次实际执行前先持久化尝试和预算。FP16 参考失败即停止；候选有效数值失败照常登记，但跳过其性能阶段。采集、工具执行或证据校验错误停止整次运行。成功、失败和中断都只允许运行一次，不删除锁续跑，也不重用已消耗的尝试。Windows 私有 Job 在放行 helper 前接管本次子进程树，退出或超时时只清理自己启动的进程。

## 读取结果

- `prepared/result.json`：本次执行是否完成、实际数值/性能尝试数、日志和账本头 SHA。
- `prepared/summary/selection-summary.json`：有限候选的结果；可能保留基线、指出证据不足，或列出仍需独立确认的候选。后者不是已选定部署计划。
- `prepared/collections/`、`comparisons/`、`performance/`：原 protobuf、数值指标和 ABBA 四臂原始计时。
- `prepared/events/`、`ledger/`、`logs/`：顺序、失败与来源记录。本地哈希链用于审计，不是外部签名。

正式选型还需要合格独立语料与 selection 分区、完整工作负载、已验证的历史基线、独立确认和一次最终留出验收。不要把 `smoke` 改成 false 就宣称认证完成；工具会检查语料资格，并仍禁止发布。旧基线的配置、q64 环境和负载差异见[主规格 §8.8](RustGo统一推理优化量化后端.md)。

collector 现提供独立的 `--execution-spec` 入口，配合 `--config` 保留旧 FP16 plan 或全 FFN INT8 的显式 q64/tune0 条件，并核验前后指纹及实际 Worker 日志。该入口已完成 B15 两路径接口验证，见[主规格 §8.12](RustGo统一推理优化量化后端.md)。它尚未纳入本页的 v1 自动种子账本；不能把旧 spec 换成 recipe 参数，也不能用旧 FP16 collection 替代比较器要求的统一完整 FP16 reference。

正式选择使用独立来源语料：`target/ptq-pro-corpus-r2/corpus/manifest.json`。校准/选型/留出分别为 32/32/128 盘、2044/2004/8049 个局面，完整合法回放与独立审计通过。`READY` 仅表示组成达标；B11/B15 首轮 selection C1 及 B11 q64 续接均已完成独立审计，完整控制矩阵没有可采用的新候选，最终留出仍未读取。B11/B15 的 CPU 搜索计划在 `target/unified-quant-qualified-corpus-plans-r1/{b11,b15}/search/`。该语料包含初始布子局面，必须使用完整 `*.requests.jsonl`；旧 `*.worker-fixture.json` 有排除项，不能替代完整分区。来源、固定分区和分布见[主规格 §8.14](RustGo统一推理优化量化后端.md)，结果见 §8.19～8.20。

catalog-v2 已接通完整 ABBA 入账及独立的 `prepare-execution` / `run-execution` 入口，用于统一方案和两个旧控制的持久尝试记录、原始采集验收、CPU 重比及完整控制矩阵汇总。不要将 v2 账本作为上面 v1 driver 的直接替换参数；两套 schema 和命令保持独立。151 项本轮相关 CPU 测试通过，真实选型结果另记；边界与证据见[主规格 §8.16～8.18](RustGo统一推理优化量化后端.md)。

v2 的准备入口为 `prepare-execution --catalog <catalog.json> --expect-catalog-sha256 <SHA> --ledger <既有持久目录> --expect-ledger-sha256 <contract.json的SHA> --expect-head <当前锚> --output <新目录>`。执行入口为 `run-execution --prepared <新目录/prepared-execution.json> --expect-manifest-sha256 <准备结果SHA>`。同一实验系列只初始化一次共享账本，不能为了重试已消费的 key 另建空账本；源码或证据变化也不能修改旧哈希绕过检查。准备只读复用已验证结果，pending/failed 不可重跑，完整剩余预算不足即拒绝准备。

v2 的 `prepared/result.json` 记录实际新尝试数和账本头；`prepared/summary/selection-summary.json` 包含完整控制矩阵；原始采集和计时仍在 `collections/`、`performance/`，数值比较与不可变证据快照保存在外部账本。Windows helper 启动前的 reservation/事件顺序由 driver 记录，单独一份旧 collection 或 ABBA 报告不能证明该启动顺序。

当前实现通过了 13 项 CPU 测试以及一次 B15 实机流程冒烟：3 个数值候选、2 组微型 ABBA 均完整记账，结果保留 FP16 对照组。该已执行实例使用 RPC 吞吐主指标，与上面示例声明的 P95 不同，不能相互替换解读。原始产物和独立 CPU 审计见[实例目录](../target/unified-quant-driver-prepare-r1/b15/prepared/result.json)与[审计](../target/unified-quant-driver-prepare-r1/b15/audit-r1/audit.json)。示例代码经过 PowerShell 语法检查，未作为第二次 GPU 实验执行。
