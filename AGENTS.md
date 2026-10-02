# Rust_KataGo

> 2026-10-02 最新本地提交与5个百分点精度交付：用户授权先提交当前代码到本地仓库，再由代理校验精度，明确采用同模型FP16基线下白方胜率最大绝对偏差≤5个百分点。代码已提交main/bdddaa4c32198868f4d76909e85fbe951e4724ce，122文件，actual0/cdc075，未push；此授权取代历史精度全由用户验证的范围。独立新阶段target/worker-original-protocol-accuracy-5pp-r1：5CPU actual0/d297ec、4case计划CPU0/a9dda1；0cdafa6f…同程序B15、同batch/C32/W32/环境，各采集128盘8049局面，4个原协议Worker均owned正常wait0、无cleanup，阶段session37651 actual0/acdc18。B3 10INT8+35FP16 FFN最大/均值/P95胜率偏差2.9831826687/0.0596581078/0.2879679203个百分点；B8 23INT8+22FP16为2.9898881912/0.0695878370/0.3311872482，均PASS5个百分点，finite/legal PASS；实际profile941b0ffc…/1e797d8f…匹配本地pin。独立raw protobuf复算session50685 actual0/b82515、32196输出、82来源首末稳定，两组最大值完全相同、4模型启动确认。最终result.json 8620B/4316e58691e19c1ef6b6e905c031185ba75c6c1708a18247c9e0f484c99bf457，actual0/43ae2d。新精度预算4/4与旧速度96/96分开，旧3锁及失败/消费/截止不变；不重启旧campaign，不认证B11新方案、FP32/棋力或其它GPU，本轮不产生新速度结论、不继承旧exe速度认证；无生产Server/Worker替换，D:/Go/Server源码不变。

> 2026-10-02 最新用户边界与已执行交付（优先于下方历史 Server 同步/typed 记录）：用户确认“Server 不需变更，只在 Rust_KataGo 推理侧动”并授权全权执行。D:\Go\Server 已恢复 b9873656b5d158664ae79dd12e7b71309881d54c 原源码，git 工作区干净；此前7个tracked改动撤回、2个新测试/示例移出并归档，result target/execution-profile-rollback-r2/result.json，actual0/5405a9。r1仅准备解析因git状态首行空白被strip丢失而拒绝，未写源码，actual1/fe875a原件保留。Worker proto 精确回原HEAD/v1；移除网络 profile 与 ACK 要求，保留内部实际 identity、本地 cudaQuantExpectedProfile、模型/配方/NN/Graph缓存及task/duplicate/cancel检查。新增6原协议CPU actual0/bc4ec7、本地pin2 actual0/37e506、preparer5子进程actual0；后者wrapper首次Windows双CR末行匹配失败actual1/30ce74，仅只读终态审查actual0/23dfc1确认5PASS，不重跑。新native15456768B/0cdafa6f90094a11dbd604642c50980ad0bc28887e5fcf2ffcc4fd85c21186ed，actual0/46f6ac、3357来源稳定。scripts/prepare_quant_worker_v1.py只做CPU quant-inspect+CUDA元数据，B3/B8准备actual0/3ff686、08bea1，预测profile941b0ffc…/1e797d8f…；新原协议B15/B8 Worker/原Pool sole实际actual0/05a53f，Worker7968/probe73832均owned normalwait0，completed1/failed0/inflight0、actual profile1e797d8f…匹配本地pin，incoming半关闭仍未知。原Server库探针只在Rust_KataGo/tools，不改Server业务。新程序只完成兼容验证，没有新ABBA或速度认证，不继承旧构建提速；精度全由用户验证。新范围模型native已96/96，不重启旧campaign/重置预算截止，旧3锁及失败消费保留，无生产进程停止/替换、无提交推送；历史生产PID当前未观测到，不能声称仍存活。最新启动为target/worker-original-protocol-preparation-b8-r1或b3-r1/launch-worker.ps1，旧typed launch_quant_worker.py封存为历史、不用于当前原Server。报告target/worker-original-protocol-composition-r1/result.json，构建target/worker-original-protocol-native-r1/verified-build.json，使用方式见docs/RustGo统一推理优化量化后端.md顶部。

> 2026-10-01 用户路径更正：Server修改已按明确要求同步到D:\Go\Server主仓库；今后该路径为Server源码/编译/使用入口。同步7个协议/实现/测试/示例源码，与首次已测worktree逐字节一致；更新主仓库已有公开协议文档及导出清单。主仓库13项execution_profile CPU检查actual0/6703da通过，公开导出工具18PASS/4SKIP actual0/f0a910，git diff --check actual0/d7a6d7。原独立worktree与已测证据保留，不将其当新的日常Server入口；历史“原Server主树未改”仅描述首次验证时状态，已由本次授权同步取代。未启动、部署或替换运行中的Server/Worker，未提交或推送；新范围模型启动仍95/96。回执D:\Go\Server\target\execution-profile-sync-r1\result.json。

> 2026-10-01 最新速度范围交付：统一 CUDA/Worker 入口与 typed execution_profile_id 已完成实际接线；新增 Server13、loaded identity7、Worker6、Worker入口7项CPU均PASS。新CUDA构建9a1f42f7…单独8样本ABBA，session94713 actual0/582f57，B15/B3、B15/B8吞吐相对最快已测旧基线提高1.787332% / 1.583087%，只支持本轮ABBA，无新程序独立更长确认；实际profile411a535d…/2ef91ab2…不继承旧程序认证。B11新程序64行正常完成0/b84ef4，不构成提速；无已确认B11新方案。统一入口scripts/launch_quant_worker.py默认prepare、按真实身份选已测配置或显式原后端回退。两组真实Worker/Server组合session44392 actual0/27272f、4进程正常0、completed各1/failed0/inflight0；incoming半关闭未知保持，正常Worker退出依据owned wait。新范围累计95/96次native启动，三旧锁再次原SHA一致；旧campaign/失败/消费/截止及生产Worker/原Server主树保持。精度全由用户验证；非RPC/搜索提速、物理batch分布/多模型/其它GPU认证。报告target/unified-quant-typed-worker-speed-r1/run/report.json、组合target/unified-quant-typed-worker-composition-r1/result.json，运行方式见docs/RustGo统一推理优化量化后端.md。

> 2026-10-01 目标继续执行：统一本地入口 `scripts/launch_quant_inference.py` 已完成8项CPU检查和B15/B3、B15/B8两次真实推理，actual0/25e8e9、actual0/773e2c，实际profile匹配既有已测配方；结果 `target/unified-quant-runtime-launcher-r2/result.json`。这两次仅验证入口身份与正常完成，不新增加速结论；确认提速仍来自前轮独立ABBA。新范围累计84次native启动，前轮已到期测速计划封存不续跑。Server typed profile与Search生命周期隔离在独立worktree完成13项CPU检查actual0/7ab7a0，未启动Server；Rust Worker接线和新构建速度验证仍在推进。目标active，精度仍全由用户验证，原campaign/失败/锁/消费/截止与生产Worker保持。

> 2026-10-01 速度范围本轮终态：48 个配置初筛、5 组正式 ABBA、2 组独立更长 ABBA 确认完成；正式 session87418 actual0/42b281，确认 session15081 actual0/3b2e02。RTX 5070 Ti / C32 上，B15/B3 10INT8+35FP16 FFN、B15/B8 23INT8+22FP16 FFN（均 q64）确认吞吐相对最快已测旧基线提高 1.844883% / 1.753228%；B11 三组未确认提速。新范围共82次native启动、80正常0、2失败原件保留，无精度检查；旧campaign/锁/消费/截止未重启或重置。最终结果与配方环境身份 `target/unified-quant-speed-only-final-r1`；详见 `docs/RustGo统一后端速度验证.md`。计时覆盖NnEvaluator请求链，排除加载/预热，非单请求P95/RPC/搜索速度；SM120实测仅5070Ti。IMMA算法标志已观察，硬件利用率因ERR_NVGPUCTRPERM未测；正式与确认独审完成，生产Worker未替换。

> 2026-10-01 用户后续明确“继续执行吧。重启目标”：已按速度范围恢复实际工作，使用独立有限性能测量入口，不重启旧单次 campaign。当前入口 `target/unified-quant-speed-only-r5`，复用既有模型、配方及其兼容程序；旧精度/holdout门退出代理验收。此前“暂停”是历史状态，最新工作状态与速度报告以此新入口为准。

> 2026-10-01 用户目标调整（优先于下方历史验收要求）：代理只验证统一推理优化量化后端是否提高实际推理速度；全部模型精度验证由用户完成。代理不再执行输出误差、policy/胜率/目差、棋力、selection 精度门或 holdout 精度验证，也不把这些结果作为性能测量的前置条件。复用已有模型、量化参数和候选配置，以同模型、同输入、同 batch/并发条件下的现有后端为基线，报告端到端延迟、吞吐、重复测量波动和加速比例。保留真实执行、模型/配方身份及正常完成检查；速度结论不表示精度通过。旧失败、锁、预算、消费及原件保留，新范围不授权重启旧单次入口或重置旧额度/截止。当前实验保持暂停，本次仅调整范围；后续按独立性能测量入口实施。

> 最新§8.104：独立闭gate诊断首次freeze0/f715c6、2新源AST通过；首次owned actual0/2331ce、0.406s，outerhelper61248/venv22184正常0、Job0，16外层/8内层来源稳定。两即刻成员列表均仅helper63196，held PID/birth/image/Job及终态cleanup3758096385通过；16ms父locator后最终Jobtotal2/active0，第二成员仍UNKNOWN，不能解释或改写旧失败。plan9533b59f…/result2d0005de…/probeabb7ff69…/root0b6234f9…。原gate/sole-process门不改；仅增加≤2s/10ms/≤201次晚成员观察的新r2源码准备中，r1不重跑。wholeconsumer资格/真实history/productionledger/GPU/ABBA0，原失败/五数值/15与16消费/总钟保持。

> 最新§8.103：wholeconsumer一次性计划首次freeze实际0/08e484，6新源AST+compile PASS、2962来源首末稳定、生产stock精确2949。plan1435705B/ec164070…、freeze37b3a17a…。首次实际session69622终态1/0d2f03、34.719s：configure返回后首方法prepare在gate前因Jobactive2/total2拒绝，1error/其它3未跑；正常phase waits/ready/chain均未知，cleanup3758096385/Job0，第二成员身份未观察不得猜conhost。outerhelper56740 actual1、helper回执ownedvenv42492 actual1，outerreceipt的owned null不补改；2964首末来源稳定。result9c6bc305…/unit e44e68f3…/root ec9ce497…。原失败与新失败不重启、无whole资格；闭gate单helper成员身份诊断源码准备中。旧三锁SHA再验一致/businessclaim无，新history/productionledger/GPU/ABBA0，原五数值/15与16消费/总钟保持。

> 最新§8.102：bootstrap首次owned freeze实际exit0/38d8bc、4.781s，31外层来源稳定，helper37252/venv59584正常0、Jobactive0,total5。原2929库存全保留，2947静态首末实读稳定/entry总2949；bootstrap720146B/b0ca4432…、entry22992B/940dfc59…、freeze-record1430694B/0ab452e5…，唯一literal替换及无新输出Sourcecycle已核。plan72c468d4…/result00369cfa…/root8af9ec4b…；仅构造成功，generated entry未运行、wholeconsumer未资格。真实OS/finalACK-release与隔离业务fixture源码正在实现。原失败/五数值/15/16消费/总钟/旧锁保持，新history/ledger/GPU/ABBA0。

> 最新§8.101：held-chain r1首次actual1/cd6021，live两父边核验通过但退出后image查询WinError31，1error/2未运行，旧inner/venv正常wait未知保留。新shared held_terminal从同retained handle读取PID/birth/exit/code，lifecycle-r2/controller-r3实际派生；r2三项首次CPU实际PASS/exit0/dd2cb1，3PASS/0skip、25来源稳定、6held wait均0，outer helper79740/venv32304均0/Jobactive0,total20。resultcaf54ea5…/root853c27a7…，旧失败不回填。两changed consumer+新freezer首次3AST actual0/dd9f54；final9 deliveryc3b85938…、bootstrap request21b36169…已冻结，bootstrap尚未生成/完整consumer资格未运行。原失败/五数值/消费/总钟/旧锁保持，新history/ledger/GPU/ABBA0。

> 最新§8.100：entry/bootstrap、held venv链、controller/lifecycle与真实terminal validator已实际接线并联合静态审查；最后ACK退出竞态、held接管超时清理和资格终态关联已修。8新源码首次AST+compile actual0/1c4b52、10来源稳定（result f530d2b4…/root14aa19e2…）；r1固定后仅派生controller-r2补qualification helper末核，单项首次actual0/b96040，其余7不重跑。新controller8d99b474…，delivery281e82c3…/16来源稳定。全为语法/静态证据，未import业务、生成bootstrap/entry/业务plan或完整CPU资格；原gate+真实venv的held-chain3检查正定义，未运行，旧IPC3不重跑。旧3锁不变、claim/运行目录无，新history/ledger/GPU/ABBA0；原失败/五数值/消费/总钟保持。

> 最新§8.99：RemoteOwner/adapter、完整facade、driver实际接线源码联合静态复核无新增确定阻断；首次lease前准入顺序已修，Prepared字段/原R/旧锁/消费/截止保持，C8追加独立C3 comparison不重采。guard9099abdf…、adapter34d2567b…、map54996074…、facade07e320fc…、driver8223bbb0…、deliveryf6c76cc0…。首字符串派生重复匹配exit1/fe34b1在发布前拒绝，保留后限定函数派生actual0/c4b3cf；非消费者试跑。新组成未import/AST/test，entry/controller/venv held链/实际终态validator和整体资格待完成；guard5/IPC3/routing4不外推。旧3锁SHA再验相同，claim仍无，新history/ledger/GPU/ABBA0，原失败/5数值/消费/总钟保持。

> 最新§8.98：真实IPC三项首次actual0/f6d12c，3PASS/0skip、20来源稳定，controller14716/10.688s，helper41680/ownedvenv7716正常0、Jobactive0。144080B双向一致，真实pending read CancelIoEx→completed995→release；wrong birth/Job精准拒绝、child hello109。三direct child11764/46272/65580实际0、各Jobactive0；实际total每phase2/outer11不猜原因。独审21小结果/日志稳定。result0f396264f97f4f2dd2aef084f930cc42028c72b876a33ee9383f8c5eb9fd139a，root71e512ffb042b0e0a0b08fecb60b101cc69b64fee96544cb2834027740e54f0e。通道资格不等于业务；Owner/facade/driver新接线进行中，claim/ledger/history/新GPU/ABBA0，原失败/五数值/消费/总钟保持。

> 最新§8.97：路由拆分首次4项CPU actual0/fb6e7a，4PASS/0skip、20来源稳定；controller53364/0.375s，helper83604/ownedvenv34708正常0、Jobactive0。保留原完整active契约及锁内/写后fullR；仅外层路由A4→3、A5→4，原参数不变，观察时序差异明确、未测收益。result2e4ae00e53a872f78b1bf54d1d450b59dd5898f7778531e3f16e6e29b70aa9d5，root2f2c07606c4565b417ce72d2b42d8f666503edba251e61b0502af7a1cb56fba5。与前两节累计13项首次CPU完成；通道静态完成，真实IPC3与Owner/driver/facade接线待完成。原失败、5数值、历史消费、总钟保持，业务claim/新GPU/ABBA0。

> 最新§8.96：接续guard首次5项真实小文件/合成边界CPU actual0/835d12，5PASS/0skip、24来源稳定；controller40060/0.875s、helper18808/ownedvenv39440正常0、Jobactive0。封存前独审修复ASCII事件摘要、close成功首末Source/claim、create_claim和boundary末核后截止；正例及写后失败保留覆盖，无新增fullR。guard d3c3132a…、map3aaaecda…，result16e623e7f5f0632fac6ece164439a2e6fd0c2ac132cfe53b77e8acfb180da2b9，root852fde1d13a106b04d450941ff6ace3281d774da8e087d275c78aa4ee43960ef。与§8.95共9项首次CPU完成，不是新campaign资格。真实claim/锁转移/ledger/history/GPU0；认证child通道、driver/facade接线及owned组终态仍未实现，原失败/五数值/消费/总钟保留。

> 最新§8.95：全部11行暂缓的矩阵阶段首次4项helper CPU actual0/f8cebf，4PASS/0skip、18来源稳定，controller0.406s，helper15620/ownedvenv67864正常0、Jobactive0。新阶段保留11事件、逐行local检查和原finalization，full replay由11次改为前后2次；明确是新观察时序，不声称旧中间全量观察等价或实测提速。独审所见逐行截止/先发布outcomes问题已修，同一active/deadline不延长。result35efebf925566511c2f0f838b22407fa5c5eb4181540866026bf5d41304730ac，root70e83121130001b4c077d98fa91a2a6622dc7c848d56a6225fc5f02ab44d035d。仅helper真实固定元数据+回调替身，driver仅AST，实际新consumer/claim/history/ledger/GPU均0。接续guard仍静态复审中；原失败/消费/截止/五份结果保留。

> 最新§8.94：路径祖先遍历候选首次6项CPU差分PASS/actual0/1f054e（15来源稳定），不删metadata查询。原完整contract无profiler固定预热+A/B/B/A首次session98051 actual0/820cb6，5×81,661条摘要全同；A1/B1/B2/A2为14.9721/14.7272/14.9604/15.2801秒，描述性均值比1.0190、A漂移2.036%，不采用、不声称解决整轮超时。result6b01b8cb…，诊断cd567b72…，根退出7b08a9bc…。另从原会话c605fc找回55156→47808→47840同期PID/命令链，片段提取67770e8c…，未补造CreationDate/inner直接Job查询。基于19固定输入的所有权纯准入首次4CPU PASS/actual0/781630，31来源稳定，result014228688c10c0995df5fea86df6339c4015cf4fc0cb0f652e15a00e65374e16、根退出dfe1ddbfe49f41fc3c0efb51da8d72f1008cf0c3d7b1a8e587d9c8c8ef521133。仅静态证据资格，旧3锁/失败/消费/截止保留；真实successor claim、driver/facade接线、全量replay及剩余GPU/ABBA未执行。新单outer所有权协议静态实现中。

> 最新§8.93：固定812来源/55,110,863B的S-P4-P4-S诊断实际exit0/456a49，四阶段212/238/230/284ms，描述性均值比1.061但串行首尾漂移28.88%，不采用并行、不声称全量或推理加速。随后原`_records(contract)`单次cProfile r2首次session41698实际exit0/c97218，controller22.078s，helper17416/ownedvenv31208实际0、Jobactive0，16元数据来源稳定；81,661条结果，带profile耗时21.4719131s，其中path_key累计21.1164609s、nt.stat989,176次。result e361c4520b792e66775f538f77ac4cca20cecce21f07c6c064c828ec1e1ff196，diagnostic159e79a63aa92ce80f1bf75b38df36bdb91892f7bd3721ccdc77bb91811ce0e9，根退出4e77bb81c56a1aaf07aedc8788ce72e07132b685de570602c599ad3287950c38。仅定位清单路径检查，无完整内容核验/history/ledger/GPU。profile r1冻结后静态审查发现计划二次读取未绑raw和空profile失败报告问题，未执行；r2隔离修复首次运行。原selection失败及5份结果保持；partial选择器仅定义，新namespace被Python/native父证明硬编码阻断，尚无新可执行消费者。

> 最新§8.92：新隔离facade仅两处Source循环接并行phase，原parent/history未改、旧pins不放宽。首次4项真实小文件/合成phase接线CPU actual exit0/chunk4d028a，4PASS/0skip、controller0.5s，helper11904/ownedvenv81312均实际0，Jobactive0，20绑定来源稳定。result4d59f6a1c449db7fcedc5f148da568f4901af016100c2eea8cf978176cf7d544，根退出b84a82c9f516c4834a1e90f67de6510c75afa000b3c981347d99dbd2d38caee5；source-map97342c80…。原8项不重跑，非真实历史清单/读取提速/新campaign资格，尚未接入业务。共享旧FP16转C8的5已完成+5+6+4=20静态分析8e76f16d…已保存，原global28h及后组6/8/6h不重置，尚无新消费者/运行时间投影/登记。旧selection失败、全部消费、新ABBA0保持。

> 最新§8.90部分终态独审：PASS_PARTIAL_TERMINAL_EVIDENCE，实际只读审计exit0/e4b8a1，报告target/unified-quant-selection-b11-c3-partial-audit-r1/report.json，109885B／b35610e9534d8b83e0b4140b9a08055b5ec70920c284abf30ca36d2140c1d449。204项有限检查通过，v4 66–81与driver0000–0020局部链、6个Prepared命令绑定一致；5预约/5采集/5数值PASS（含参考自检），005旧FP16无发布预约/启动事件/目录，CPU45无result。42证据及审计实现稳定；AGENTS期间更新已如实单列变化，未冒称全43输入稳定。3锁保留、新ABBA0，原整组exit1不变；非81619全量末核/raw复算/性能/holdout。§8.91并行核验8项CPU已PASS，最小两循环接线在新隔离目录静态制作，尚未实际调用方验证或新GPU。

> 最新§8.91：并行Source核验原型首次8项合成CPU检查actual exit0/chunk6a25c0，8PASS/0skip，controller0.953s，helper35540及owned venv69616均实际0，Job终态active0；15绑定来源稳定。result c95d3de0773dee8eeb177b39379c1f726c72918ce94a00cda345818dbda4ecc9，根退出30e97836a0ce501ed0b143a08120b62ee80859d480075f154fbf4d46d8a22d44。仅并发/快照/异常/截止/join合成行为，不是实际Source回调或提速资格，未接入业务。独审指出独立CPU适配器最后写入后缺截止复核，r2隔离补两处同一绝对截止检查（change7459e89b…），未执行、原8项不重跑；原r1已测成功原件保留。B11/C3原session28076失败终态与5份数值结果、新ABBA0不变。

> 最新§8.90：原session28076已实际exit1/chunk68b5cb，北京时间2026-10-01 03:48:07；controller终态FAIL_STOPPED_NO_RETRY，绝对work deadline/watchdog触发，run Job终态active0，whole_descendant_exit_certified=true。正常helper/venv退出仍null，cleanup_exit3758096385，不能补写成功；controller来源末核稳定不等于被中断CPU45的历史全量末核。root实际退出观察f337d878…，controller result83b6b378…、run-exit def16e72…。参考、3混合、旧INT8共5份数值结果保留；旧FP16 CPU44 PASS、CPU45 reserve有started无result，当前事件止81，未见82或旧FP16采集目录。部分账本独审进行中，新ABBA0/其它3组未执行。禁止重启原入口或重置时间/消费；success-only下一组freezer不能消费该失败。并行来源核验隔离原型8测试仅定义，独立CPU资格准备中，未接入运行链。

> 最新§8.89续更：旧FP16控制CPU44 check已PASS，result1290B／0ed3692fc6d0b3cff66c1dbf9d6a32f3e0bedfbbc4b6b9bed393aaa6f4dea956，inventory_checks=1/history=81619/source_set9904…；CPU45 reserve操作已开始，started55a6feba…，第82事件及实际启动尚未观察，不构成已确认的永久预约消费，新预约仍5/ABBA0。原session28076轮询7a594d仍LIVE，随后时钟UTC 2026-09-30 19:44:16、metadataf87bed；root观察18d3ffcf…。独审仅CPU44 result/started及CPU45 started三小文件稳定、绑定匹配，未审第82事件，不证明永久预约已消费、新OS退出或整组终态。继续原会话，不重启或改消费与截止。

> 最新§8.89续更：第三混合比较后CPU43 check已PASS，result1290B／f5d08ae88255993f20a6ff03c221b975640b696fc882342fa93199927a07a8b6，inventory_checks=1/history=81619/source_set9904…；旧FP16 b11-legacy-fp16-c32／005-collect的CPU44 check已启动，started875cc5c9…，未见第82预约事件、实际启动或整体终态。原session28076轮询437ed0仍LIVE，随后时钟UTC 2026-09-30 19:36:09；root观察05820b25…，独立CPU43 result/started及CPU44 started三文件稳定、绑定一致，未跟读引用/raw，历史exits不作新OS退出。参考+3混合+旧INT8数值通过保留，新预约5/ABBA0保持，继续原会话、不重启或改消费与截止。

> 最新§8.89续更：原session28076轮询8cc0d5仍LIVE，随后时钟UTC 2026-09-30 19:27:28。第三混合CPU42 compare实际PASS，result1292B／619ed4a5e04287466e579baeb61c3652e594e0a18c36bd2ba0e5ba07a4b7d04e，inventory_checks=2/history=81619/source_set9904…；driver0020 NUMERIC_VALIDATED、gate PASS／d8225cb3…，绑定第81事件head92f89418…及comparisonc6c45…。CPU43 check已启动，整组未终态。root完成观察2278bde1…，独立五文件有限复核稳定、链与绑定匹配，未读raw/引用或CPU43，不证明新OS退出、整组或性能/holdout采用。当前参考+3混合+旧INT8均有通过数值记录，旧FP16待完成；新预约5/ABBA0保持，继续原会话、不重启或改消费与截止。

> 最新§8.89：原session28076轮询595adc仍LIVE，随后时钟UTC 2026-09-30 19:19:18。第三混合b11-c3-mix-003（11 INT8/22 FP16 FFN）的v4第81事件NUMERIC_COMPARED已生成，9f290e11…／head92f89418…，32盘2004局面DIAGNOSTIC_PASSED/PASS；胜率差max/mean/P95为2.1202921867/0.0576440670/0.2727955580个百分点（门限6），最大目差1.40966796875目，Top1 1987/2004（99.1516966068%），finite/legal PASS、exact0/2004。root观察148acb63…，独立第79/80/81三事件前后稳定、canonical与身份匹配，未独立复算raw；非自比较标志来自79。CPU42末核、driver0020及整组result尚未见，事件PASS不等于操作终态或性能/棋力/holdout/生产通过；旧FP16未完成，新预约5/ABBA0保持，继续原会话、不重启或改消费与截止。

> 最新§8.88续更：第三混合候选CPU41 ingest-collection已PASS，result1302B／99c73c3aa9178c8fc14ccaedd1f65b4a06955d61fc99865e5d778b5991e38b8a，error=null/inventory_checks=2/history=81619/source_set9904…；CPU42 compare已启动，startedccbed92f…，数值结果尚未观察。原session28076轮询1c9535仍LIVE，随后时钟UTC 2026-09-30 19:08:20；root观察c1adcd50…、metadatafd261f。独审仅CPU41 result/started与CPU42 started三小文件前后SHA稳定、绑定匹配，不把历史replay退出当新OS退出，不证明数值、性能或整组终态。新永久预约仍5、ABBA0，继续原会话、不重启或改消费与截止。

> 最新§8.88续更：第三混合采集后CPU40已PASS，result1290B／f8dafb2381c2c062892d728fb70639a0355f533969e07f469a9213dc97c17928，inventory_checks=1/history=81619/source_set9904…；root检查观察5d7c4470…。原session28076轮询0b5b27仍LIVE，随后时钟UTC 2026-09-30 18:59:56；metadata719f6a确认v4第80事件COLLECTION_OBSERVED已写入，33198de1…／head d1202041…，validation为RAW_PROTOBUF_AND_RESTORED_EXECUTION_IDENTITY_VALIDATED、numeric_gate=NOT_COMPARED。独审仅event79/event80/report三文件前后SHA稳定，canonical链与身份、report/profile/registration/child-exit Source一致，未跟读引用/raw；root事件观察0f7a9654…。此核对不证明CPU41末核、数值、性能或整组终态；CPU41 result和第81事件仍未见，新永久预约仍5、ABBA0。继续原会话，不重启或改消费与截止。

> 最新§8.88续更：第三混合b11-c3-mix-003真实采集已完成，native PID46672由supervisor持有并观察exit0，driver0019 PROCESS_EXITED returncode0；32盘2004局面、668×B3、4009输出文件/41,789,924字节，实际profile7cac2d6e…匹配catalog。report仍为COLLECTED_UNVERIFIED，数值/性能未通过、整组未终态。原session28076最新轮询4e9e11仍LIVE，随后时钟记录北京时间2026-10-01 02:50:33；root采集退出观察74127efa…，独立driver0018/0019、native process result、report、catalog五文件有限复核稳定，仅核链、身份与计数，未复算raw。保留原启动记录及参考+两混合+旧INT8既有数值结果，继续原会话、不重启或重做已消费入口。

> 最新§8.88：北京时间2026-10-01 02:36:20，原session28076轮询d54e9b仍LIVE。第三混合b11-c3-mix-003的CPU39启动前检查PASS／cd418c12…，driver0018 ATTEMPT_STARTED／998bd774…；native launch记录PID46672，root既有精确PID观察确认parent63824、出生时间及镜像匹配冻结exedd665d03…，root不持native handle、无退出证明。claim241f4e60…及supervisor attempt f04a7ec0…均CONSUMED_NO_RETRY。root启动观察146f6907…，独立五文件有限复核稳定；实际启动依据launch+CIM，driver事件单独不证明Popen/GPU执行。最新metadata0ac119未见native退出/report/stage/driver0019/CPU40及整体result，新永久预约仍5；GPU推理完成、数值、性能及holdout未认证，参考+两混合+旧INT8数值保持，继续原会话、不重启或重做已消费入口。

> 最新§8.87续更：北京时间2026-10-01 02:23:35，原session28076轮询493391仍LIVE。第三混合候选的CPU38 reservation-head已PASS，result1301B／c85276159e15fb053f946e9b62331823d535b4b996c12587b1373973b44bcac6；driver0017 ATTEMPT_RESERVED／e2efdf42…与v4第79事件及reservation12e39113…绑定，receipt为RESERVED_BEFORE_EXTERNAL_LAUNCH、consumed=true/retry_allowed=false/execution_started=false/publish_allowed=false。CPU39 check已启动，尚未观察result、driver0018或实际启动/整组终态。root启动前观察e1dcaff3…，独立五文件有限复核稳定，未读started原件/raw/引用、非CPU39或OS退出证明。新增永久预约仍5，参考+两混合+旧INT8数值完成保留；第三混合/旧FP16/完整矩阵/性能/holdout/生产未完成，继续原会话、不重启或改消费与截止。

> 最新§8.87：北京时间2026-10-01 02:14:49，原session28076轮询471342仍LIVE。第三混合b11-c3-mix-003／004-collect已写v4第79事件ATTEMPT_RESERVED，c3c7c402…／head d7276090…，新增永久COLLECTION预约累计5、旧消耗继承。CPU37 reserve PASS，result1292B／4b241d655b95b212494758187e6584e388be39224ecd239f826d2e4e1d89c11d，inventory_checks=1/history=81619/error=null；CPU38 reservation-head已启动、尚无终态或实际采集启动。root预约观察d530292a…，独立五文件有限复核稳定，registration与catalog对应项完全相等，未读引用/raw/started原件，不证明GPU启动或OS退出。候选B3的11 INT8/22 FP16 FFN仅为静态保存路由；参考+两混合+旧INT8数值完成保留，第三混合、旧FP16、整组/其它负载/性能/holdout未完成，29未来CPU/v5/ABBA/profiler未执行，继续原会话、不重启或重置预算与截止。

> 最新§8.86续更：北京时间2026-10-01 01:49:48，原session28076轮询19d467仍LIVE。旧INT8 q64的CPU34 compare完整返回PASS，result1292B／0df43297b45a2e5a8b10033ec48920a437e1fcbdfb2eb83cdeb123730aab6feb，inventory_checks=2/history=81619/error=null；driver0016 NUMERIC_VALIDATED、gate PASS，65f93f19…，绑定v4第78事件head964dbe10…。CPU35 check已启动，尚无整体result；root完成观察aa0a19d6…，独立五文件有限复核稳定，未读引用/raw、非独立OS退出。当前参考+两混合+旧INT8数值阶段完成；冻结下一项004-collect／b11-c3-mix-003（静态路由11 INT8/22 FP16 FFN），尚未观察其预约或启动。新增永久预约仍4，第三混合、旧FP16、全组/其余负载/性能/holdout未完成，继续原会话、不重启或改消费与截止。

> 最新§8.86：北京时间2026-10-01 01:42:14，原session28076轮询0f0bbd仍LIVE。旧INT8 q64控制的v4第78事件NUMERIC_COMPARED已生成，493900b2…／head964dbe10…；共享统一FP16参考与旧控制为不同attempt，selection数值DIAGNOSTIC_PASSED/PASS。32盘2004局面，白方胜率差max/mean/P95为2.85074413/0.10202194/0.54038167个百分点（门限6），最大score差1.05678844目，Top1 1976/2004（98.60279441%），finite及合法性门通过，exact输出0/2004、非逐位相同。root观察038ecb13…，独立第76/77/78事件有限复核稳定，未读引用/raw重算；CPU34 result、driver0016及总result仍未见，不能称比较末核或整组完成。仅selection诊断，FP32未评估、性能/棋力未测、holdout/生产未通过；新预约仍4，29未来CPU/v5/ABBA/profiler未执行，继续原会话、不重启。

> 最新§8.85续更：北京时间2026-10-01 01:32:17，原session28076轮询74cbaa仍LIVE。旧INT8 q64控制的CPU33 ingest入账末核已PASS，result1302B／6bd6c2682c36c9e0166021256e2c7c9d75df6c816bd524854c34d2a5e7930fe1，inventory_checks=2/history=81619/source_set9904b46b…/error=null；CPU34 compare已启动，started61c6e748…，尚无result、第78事件或整体终态。root入账完成观察73867ad1…，独立三小文件有限复核稳定、started Source精确匹配；未读引用/raw，CPU33 result非独立OS退出证明，旧replay两exit不能代替。第77事件numeric_gate=NOT_COMPARED保留，新永久预约4、旧消耗继承，参考及两候选数值PASS保持；整体/性能/holdout未通过，继续原会话、不重启或改消费与截止。

> 最新§8.85续更：北京时间2026-10-01 01:25:58，原session28076轮询1d74da仍LIVE。旧INT8 q64控制的v4第77事件COLLECTION_OBSERVED已写入，93789ec5…，head84d2718b…；同一attempt/reservation/execution/workload及report身份绑定通过，validation为RAW_PROTOBUF_AND_EXECUTION_IDENTITY_VALIDATED、numeric_gate=NOT_COMPARED。process_start_after_reservation仍为NOT_PROVEN_BY_COLLECTION_REPORT，不能套用restored actual_child_exit证明。root事件观察de79d028…，独立三文件有限复核稳定、未读引用/raw；CPU33仍无result，事件写入不等于CPU33终态，无数值/整组/性能终态。新永久预约仍4，29未来CPU/v5/ABBA/profiler/holdout未执行，继续原会话、不重启或改消费与截止。

> 最新§8.85续更：北京时间2026-10-01 01:19:48（UTC 2026-09-30 17:19:48），原session28076轮询848b0d仍LIVE。旧b11-legacy-int8-q64／003-collect真实采集已完成，driver0015 PROCESS_EXITED returncode0；report为COLLECTED_UNVERIFIED，32盘2004/2004请求，最终heartbeat nn_rows=2004/nn_batches=252/failed_requests=0/in_flight=0，drained=true。CPU31检查PASS917866c1…、CPU32采集后检查PASS50cd9422…；CPU33 ingest仅有started84a91e94…，尚无result，v4第77/78事件、driver0016及总result均不存在，未观察CPU34启动。root采集退出观察e3e41b9f…，独立五文件有限复核通过，未读raw或独立Worker PID退出凭据。新永久预约仍4，参考及两混合数值PASS保留；旧控制尚未数值比较或性能认证，整组未终态，29未来CPU/v5/ABBA/holdout未执行，禁止重启或改消费与截止。

> 最新§8.85续更：北京时间2026-10-01 01:01:06，原session28076轮询481db4仍LIVE。旧INT8 q64控制的CPU30 reservation-head已PASS，result1301B／53799d835599c057c07a8ff9b793ec04d05b5cf9a51fb2a09ecb98e233415fbd，inventory_checks=1/history=81619/error=null；driver0013 ATTEMPT_RESERVED／7c81570f…与v4第76完全绑定，receipt为RESERVED_BEFORE_EXTERNAL_LAUNCH、consumed=true/retry_allowed=false/execution_started=false/publish_allowed=false。CPU31 check已启动，started5cb4cc24…，尚无result或driver0014；未观察实际采集启动或整组终态。root启动前观察cb729024…，独立五文件有限复核通过，未读started原件/raw/引用，不证明实际启动或OS退出。新增永久预约仍4，参考及两混合数值完成证据保留，完整矩阵/性能/holdout/生产未完成；继续原会话、不重启或重置消费与截止。

> 最新§8.85：北京时间2026-10-01 00:52:25，同session28076轮询c18b2d仍LIVE。旧b11-legacy-int8-q64／003-collect已写v4第76事件ATTEMPT_RESERVED，d56d53af…，head b4141124…；新增永久COLLECTION预约累计4，非自比较、maxattempt=1。CPU29 reserve结果PASS，1292B／9b2d1e87dd5f17eafc867ea1b2448bd9023e743891f3f75fe2fdc24757571ee8，inventory_checks=1/history=81619/error=null；CPU30 reservation-head已启动，尚无终态或实际采集启动/结果。静态配置为cudaint8backend、batch8/nnMaxBatchSize8、capacity32/window32，实际batch分布未观察。root预约观察fccd6df2…；独立五文件有限复核通过，未读引用/raw，不证明CPU30或OS退出。FP16参考及两混合数值完成保留，整组/完整性能/holdout/生产未完成；29首次CPU/v5/ABBA/profiler未执行，继续原会话、不重启或重置消费与截止。

> 最新§8.84续更：北京时间2026-10-01 00:28:19，同session28076轮询cb1872仍LIVE。第二混合b11-c3-mix-001的CPU26 compare末核已PASS，result1292B／cd647d817684734ac3d650de6de377b10f022a49130a86d4490c9ca46695047a，inventory_checks=2/history=81619/error=null；driver0012为NUMERIC_VALIDATED、gate PASS，b51f8e00…，与v4第75事件绑定。CPU27 check已启动，started32dd7e21…，整组未终态。root完成观察c20f2c02…；独立四文件有限复核通过，未复算raw或全部来源，CPU result不是独立OS退出证明。当前完成FP16参考自验及两混合候选数值阶段，剩余旧控制、第三混合及整组/完整性能/holdout/生产未完成；新永久预约仍3，继续原会话、不重启，29首次CPU/v5/ABBA/profiler未执行。

> 最新§8.84：北京时间2026-10-01 00:20:35，同session28076实际轮询cfb28b仍LIVE。第二混合b11-c3-mix-001（3 INT8/30 FP16 FFN）的v4第75事件NUMERIC_COMPARED已生成，f1034140…，head ba3e7d9b…；root有限3事件canonical及身份核对PASS／7bb8bf，未重算raw。实际metrics gate为PASS/DIAGNOSTIC_PASSED：32盘2004局面，最大白方胜率差0.87857246个百分点（门限6），平均0.01637967、P95 0.07740855个百分点；Top1 1998/2004（99.7005988%），合法性2004/2004，最大score差0.70106506目，exact输出0/2004、非逐位相同。CPU26 compare结果及driver/controller总结果均不存在，不能写CPU末核完成；性能NOT_MEASURED、FP32 NOT_EVALUATED、holdout/生产未通过。新永久预约仍3，继续原会话、不重启；29首次CPU/v5/ABBA/profiler未执行。

> 最新§8.83续更：原session28076于2026-09-30 16:11:49 UTC（北京时间10月1日00:11:49）经chunk0bb979确认仍LIVE。第二混合b11-c3-mix-001的CPU25 ingest-collection入账末核已PASS，result1302B／7428a7e7088ebc9144d989a21441c176b08cb0ac5a43e62b1139349c102b63db，inventory_checks=2/history=81619/error=null；CPU26 compare已启动，started227B／30ead42394c921abf356cccfd7c281a0fd5af8b703238035a36bb06be0cea28c，尚无compare result。root入账完成观察d4f1a26d…，ROOT当前阶段为数值比较在途；v4仍74事件、numeric_gate=NOT_COMPARED，不能写候选数值通过。driver/controller result均不存在，继续原会话、不重启；首mix-002数值PASS保留，完整性能/holdout/生产未通过，29首次CPU/v5/ABBA/profiler未执行。

> 最新§8.83：原session28076于2026-09-30 16:03:54 UTC（北京时间10月1日00:03:54）经chunk307021确认仍LIVE。第二混合b11-c3-mix-001（3 INT8/30 FP16 FFN）native PID8544已由supervisor持有并观察exit0，launcher driver0011 returncode0；668×3=2004局面/32盘，4009输出文件/41,789,958字节，report为COLLECTED_UNVERIFIED，非数值通过。native退出观察d511d95b…；CPU24检查PASS，result4b4f2b12…，inventory_checks=1/history=81619。CPU25 ingest-collection已启动、尚无result，CPU26未启动；v4第74个COLLECTION_OBSERVED事件已生成，事件记录原始PB与恢复执行身份已验证，numeric_gate=NOT_COMPARED，head3f2dcaac…；这不构成CPU25终态或数值裁决。新永久预约3，外层及driver result均不存在，继续原会话、不重启。首mix-002数值PASS保留，完整性能/holdout/生产未通过，29首次CPU/v5/ABBA/profiler未执行，窗口内仅静态。

> 最新§8.82：同session28076实际轮询be9dac仍live；第二混合候选b11-c3-mix-001的CPU23启动前检查已PASS／e818b518…，driver0010 ATTEMPT_STARTED／2824101f…；native launch记录PID8544／5ea438ec…，精确PID查询镜像匹配冻结exe（仅观察，非root held退出证明）。claim9814d683…与supervisor attempt b084c8de…均CONSUMED_NO_RETRY。尚无native退出/report/数值或整组终态，禁止重启。root启动观察5c35075e81a8a3c78f81d2e72c2e560aa8809f4e09b9c38406f7871aed1635f3；新永久预约仍3。首mix-002数值PASS保留，性能/holdout/生产未完成，窗口内仅静态。

> 最新§8.81：同session28076实际轮询99be43仍live；第二混合候选b11-c3-mix-001已新增v4第73号ATTEMPT_RESERVED（文件fd201d61…／逻辑headaa385f94…），CPU21 reserve实际PASS／adfd0f61…，新COLLECTION永久预约累计3。CPU22 reservation-head已启动，尚无claim/native launch/采集完成或整组终态，禁止重启。候选为3 INT8／30 FP16 FFN；事件canonical摘要、冻结Prepared身份和登记Source有限核对相符。root预约观察0801b1e656c0b0856007e744fb7f6cc7ae08675eb1253e8db0de587f9b641ff7。首mix-002完整数值PASS保留；29首次CPU/v5/ABBA/holdout/生产仍未完成。

> §8.80最新续更：同session28076实际轮询99d90c仍live；首混合候选CPU18 compare末核已实际PASS／870116e2…（inventory_checks2、history81619），driver事件0008为NUMERIC_VALIDATED／gate PASS，文件50343def…，绑定同一candidate156bc8…／comparison5bbe614…／v4 head711d1fc2…。CPU19 check已启动，整组未终态，不能重启。root数值阶段完成观察419740913ff5e8817dc220f93eb80a296756ad7ba1f6eba13e3ba87c62abc57e；完整性能矩阵资格、ABBA、holdout与生产仍未完成。

> 最新§8.80：同session28076实际轮询1d0b77仍live；首混合候选b11-c3-mix-002已写v4第72号NUMERIC_COMPARED（文件5e95820b…／逻辑head711d1fc2…），与FP16参考为不同attempt，登记明确非自比较。32盘2004局面实验gate PASS／DIAGNOSTIC_PASSED，最大白方胜率差1.36823356个百分点（门6）、平均0.04054958、P95 0.20948946；Top1 1990/2004，最大目差1.704319目，逐位相同0/2004。CPU18末核及整组仍未终态，不是性能/holdout/生产通过。root数值事件观察e66f87dcd205bccbe595d6a374a57eca9535391d57ab3d04536aca9a06689389；事件摘要及三事件身份有限独立复核，未重算raw。禁止重启，29首次CPU/v5/ABBA/profiler仍未执行。

> §8.79最新续更：同session28076实际轮询aa74c4仍live；首混合候选CPU17 ingest现已实际PASS，result c13bad8cd6fcef278d1168a8a8dc79d69d1ac85bd1b476b6e5bfd9ff73860ffb，inventory_checks2／history81619。CPU18 compare已启动（95166c4c…），尚无候选裁决或整组终态。root入账完成观察f0106be0112e7befd2f44978d8066d2b17034757b5dd0301fca5108cc54c2a72；只续看原session，禁止重启或将ingest PASS写成精度／性能通过。

> §8.79在途续更：同session28076实际轮询f3dc6b仍live；首混合候选新增v4事件71 COLLECTION_OBSERVED，文件f0b72ea1…／逻辑head0207589d…，前序70衔接及canonical摘要已有限复核。事件为RAW_PROTOBUF_AND_RESTORED_EXECUTION_IDENTITY_VALIDATED／NOT_COMPARED，绑定19118 runtime来源；CPU17尚无终态，不代表数值比较或整组PASS。root事件观察d30ab1040951a6ee1309bbf5b28486f5ebd7c5ced684e6bcb31c458242f95c45，未独立复哈事件全部引用，禁止重启。

> 最新§8.79：同session28076实际轮询48875b仍live；首混合候选b11-c3-mix-002的native81736由supervisor owned handle观察exit0（97e3f88e…），launcher事件0007同为returncode0（aa1d7a1f…）。668×3共2004行、32盘、4009文件，stage为COLLECTION_COMPLETE_NOT_ACCURACY_OR_PERFORMANCE，report4643c10b…仍COLLECTED_UNVERIFIED；不能据此称混合精度通过。采集后CPU16实际PASS／91732bb0…，CPU17 ingest在途，尚无比较或整组终态，禁止重启。root有限观察25a832a04385cd8d1a6532802d592a650090210e8fc2c8be232353bced2541de；未重读raw或复哈全来源。29首次CPU、v5、ABBA及profiler均未执行，当前窗口只静态。

> 最新§8.78：同session28076实际轮询25357e仍live；首混合候选b11-c3-mix-002已写ATTEMPT_STARTED（driver0006／729fcd70…），claim为CONSUMED_NO_RETRY，native launch记录PID81736／d6576610…；尚未观察native退出、候选精度或整组终态，禁止重启。root启动观察1d43d766eb42a3514688d2e1fe074c9beb5261555fdb07053e3c90c5556c85d2。有限只读包路由观察994c360c…确认B3 rows1083、6 FFN INT8/27 FFN FP16、包6450ec00…及内嵌目录原文SHA7df28fb3…；源码明确目录是expected routes，非逐kernel执行trace，不能据此宣称Tensor Core利用率。官方Graph profiling与动态指令证据边界补入§9；未运行profiler/query或新增预算。29首次CPU仍未执行，窗口内只静态。

> 最新§8.77：原session28076实际轮询a556ae仍live；首个混合候选b11-c3-mix-002已新增v4第70号ATTEMPT_RESERVED（文件8564a16b…／逻辑heade2ac9b36…），reserve CPU实际PASS／052f28c3…，新COLLECTION永久预约累计2；cpu-000014-reservation-head在途，未观察候选采集完成或整组终态，禁止重启。root观察085f00931b0b28e2209c8306b9ccc68d329d78a385431078400d6b1fb007453e。性能contract容量overlay静态封存1d276cb7…，消费接线db886463…／caller描述66e2e5ef…，合并独审ad8c1372…；完整原pretty字节≤128MiB与截止均在双seal前，写同raw并核snapshot Source。首次资格现4/15/4/6=29，library新cpu-r3、caller新cpu-r2；所有首次根及旧未执行根均仍不存在（77e954），无测试/AST/import/freeze/v5/ABBA。仅静态修复及独审，不证明未来实际容量或资格。

> 最新§8.76：同session28076实际轮询a8eb0d仍live；首B11/C3参考ingest与compare CPU均PASS（d28887bb…／d20d76e3…），v4第69号NUMERIC_COMPARED文件3cbecd6a…／逻辑head901074fc…；明确是同一参考attempt自比较，2004/2004输出一致、32盘、误差0，不是混合候选或性能PASS。driver事件0004为NUMERIC_VALIDATED，当前cpu-000011-check已启动、整组未终态，禁止重启。root观察6ac5ed37c9897bfeb2c334b46cb529903f95f58a0d2d432746f4e46cd8c01308。最终性能plan assembler静态封存delivery2c52746f…／独审81f8d721…；4/12/4/6仍未实际运行，未装配/v5发布/ABBA。完整contract须在两永久seal前精确检查128MiB，最小独立overlay正在静态修复发布顺序；不改旧封存原件，窗口内只静态。

> 最新§8.75：同session28076实际轮询6170ab仍live；首B11/C3参考已写v4第68号COLLECTION_OBSERVED事件，文件e7c27cef…／逻辑head7ba4a7d6…，原19143 runtime来源及RAW_PROTOBUF_AND_RESTORED_EXECUTION_IDENTITY_VALIDATED／NOT_COMPARED保留。root观察0c36a8480d61bbb61f9ff7b95f2cf42b92fbc35c421b24423328097c43156c4b，event逻辑摘要已核；ingest-collection CPU末核仍无终态，整组未结束，禁止重启。最终performance plan assembler在独立static-r1准备，必须消费四组真实终态和4/12/4/6实际CPU证据，再物化98树与精确library/campaign字段；不伪造未来资格，无默认时限，不启动history/登记/ABBA。窗口内仅静态。

> 最新§8.74：同session28076实际轮询829535仍live；首B11/C3参考native64944及launcher均exit0，668提交/2004行/32盘，report5fa25a0a…仍COLLECTED_UNVERIFIED，driver PROCESS_EXITED事件8a5e0666…；root观察ac90eee43205b75485fe39b551673960539c82407c37fa319b6f2ba49a034e29。尚无数值入账或整组终态/精度性能认证，禁止重启。10456来源回执规范化映射相同，仅1处根路径大小写不同，root未重哈希全库存。entry r2、库12、caller及matrix4/caller6 freezer均已静态封独审；caller6只用delivery-r2 bad26646…（初1Source错误回执保留，修正30来源），原code0284a98c…未变。4/12/4/6首次检查均未执行，无v5发布/ABBA，窗口内仅静态。

> 最新§8.73：原session28076仍live（c09e3b），register-catalog及首reserve CPU操作实际PASS；v4新增000067 ATTEMPT_RESERVED，首B11/C3参考b11-c3-mix-000已永久消费，event文件SHAd41451da…／逻辑head4d7eab02…；root观察bd9192b5eeaf74ec3f62e8877d96666eaf9a1df9b9193c0cfc507592a521161b。尚无采集完成/整组退出，严禁重启。root静态核出新v5长phase直接传旧history的1800秒上限会拒；新entry-only更紧parent截止overlay在制，outer总钟不变、旧库不改。库首次CPU拟12=6policy+4fixtureoverlay+2截止检查，旧pure10从未执行。matrix完整flow2静态封存8532b4d3…，将只在caller组合suite运行一次；caller与12 freezer仍静态。工作投影0672ffd3…保留176Worker/128指纹上界，不是OS总数或新额度，窗口内仅静态。

> 最新§8.72：原selection session28076仍live（3229d8），run历史owned60908 actual exit0、receipt SHAbb7c631a…；原driver已写RUN_STARTED事件ca3fae75…，尚无采集完成或整组终态，禁止重启。root观察02 SHA431676329f913b148928eb34aba16cba0a0cbc947f0b25caa1fdba2f6fe2b6a5；01措辞更正保留，精确PID查询不是held句柄证明。v5性能续接库静态封存delivery39f14ec3…/reviewba29b80c…；10检查仅定义，generic预载冲突由独立测试overlay0cc9f79b…/review8829ed77…静态关闭。新44矩阵driver静态封存delivery3888764b…，4检查仅定义，复用原benchmark.main/reader/summary；函数返回不冒充OS退出，完整run账本流仍未测。两阶段Job caller及pure10 freezer在制，未v5登记/真实锁/新ABBA，当前窗口仅静态。

> §8.71在途续更：同session28076首组prepare已实际PASS，helper50716/ownedvenv21076均exit0、1120.469s、Job Active0；prepare-exit SHA2c5202be28029dc6ecbe70249a4b0537e73772a3f1ebe4b37c25cfeb05c28e16。Prepared SHAb86616f75c6e078e3feeb9dddd70a8664cd0e2ca24a33d55093030de2b7dad5b，6 COLLECTION/0 ABBA及11 deferred矩阵。现run helper55156/venv47808/inner47840已启动，主producer23464未终态，禁止重启；仅续看原session，窗口内仅静态工作。

> 最新§8.71：原selection session28076仍live，prepare历史v3 owned16652/v2 owned71148实际exit0，replay-receipt SHA8dda9ff8af14e46e7a3156f99398d4f0c230cebfc40d73e1b446063562ba5ed1；这不是prepare/整组终态，禁止重启。后3组freezer源码已封存独审（deliveryb569112a…/reviewd38c105c…），只接受即时前组真实root与原总钟，未执行。性能Source投影98源增量已封独审delivery0fe9b648…/review9acbbff0…；新四检查freezer仅静态封存5564f919…，未AST/import/freeze/tests，不能称实际bootstrap/ABBA资格。v5仅性能续接静态在制，不改旧v4或写真实锁/账本。当前窗口只静态并行。

> 最新§8.70：首组freezer同session32450 actual exit0/e09693，15344输入来源稳定；target/unified-quant-selection-b11-c3-driver-r1计划SHA4f40b8b801cb7f3e1b933c994ee98381b73b7bd889e4a3a3375999c735d09061，freeze-result7f56a39f…，control15349/runtime2929。首组B11/C3已唯一启动session28076/producer23464（初b3ddd4、精确命令031940），尚无终态，只续看同一session禁止重启。原登记driver根mixed-prepared-r1/driver-roots/b11-c3；history1800、四组28800/21600/28800/21600、总100800秒含组间间隔。仅预登记20 COLLECTION/0 ABBA，按实际原件判断消费；窗口内仅静态并行。后组续接freezer与性能Source投影在制，完整数值/性能/holdout尚未完成。

> 最新§8.69：最小control r3资格消费者已封存独审，delivery3cd061b6…，原Job/gate/预算/整段执行不变。新四项CPU首次direct actual exit0/b458e8，2748来源稳定、334→334imports、4PASS/0skip/forbidden；result98463891c23a3effa3c837120872fc104896b4815097e1820644eea8f66ca922，root3797c4fe0b9536359e0e5d39d6fbfafd90c6762b6ab6f3a3df2306cd339a8f61。只读实际旧局部证据与内存拒绝检查，不重跑real7/Job/模型；旧聚合FAIL保留。首组plan freezer正在绑定原登记driver根与实际pure4，尚未freeze/启动selection；20 COLLECTION/0 ABBA未消费。

> 最新§8.68：仅breakaway全成员身份诊断首次direct actual exit0/2f3553，2786来源稳定；result27bbe89b21c3c5d13684775e8a9968e63cade3b57a400da4f02661dbc177658a，root62d3d59bab416793ef6c158d5fc325b55f0b74d0b63ba6c1cfcfd288d4e38c62。本次七成员=四Python角色+三conhost.exe（a29c38fe…），extra9128/52044/58404的held出生/镜像/父身份均核对；两Job全成员true，helper/venv/leaf双wait全0、最终empty/Active0。EOF继承未重跑；不据本次回填旧r1三PID或旧Total40身份。r2诊断仍production_control_qualified=false；最小control r3资格消费者与首组plan freezer在制，旧Job/gate及聚合FAIL保留，不重跑旧real7。v4剩余20/0未消费，selection GPU/ABBA/生产仍未启动。

> 最新§8.67：两case成员诊断首次direct actual exit1/52b6f3，2751来源稳定，result4c58d1f2bcf45b742f1d6260413203a2374f16a2502f5fc7954f1a57f9621ab9，root484728b26a0893312f23c85f903ca656149d8cced7c8dfc16a33905f9ec8203d。EOF预期拒绝/actual1/两Job empty PASS；breakaway四声明身份在两个Job均成员true，但完整live列表和Active均7，额外60888/65496/67028身份UNKNOWN而FAIL。不是仅TotalProcesses重复显示；原5进程拓扑假设不能作完整OS总数界。leaf双句柄cleanup实际3758096386、两Job最终empty/Active0；不重启。后续仅新一次breakaway全成员句柄/出生/镜像/父关系观察，EOF不重跑，生产23门不改、GPU/ledger新增0。v4剩余20 COLLECTION/0 ABBA保持，完整选型未完成。

> 最新§8.66：v4同session71940实际exit0/f4434e，2606来源稳定、1675.343s；result3e7dc0d1…/root881aae5c…，实际发布结果6b76f60d…。原账本新增restored-selection-v4，head cb54af140c21dfc0ff6ac194d8d7abb086aabadec145e8835c973c19d351f04f，继承10 COLLECTION/16 ABBA，剩余20/0、未新预约或GPU，不重跑history/publication。整树CPU首次session39597实际exit1/ed019b，5pure+6real局部ok、最后EOF计数2≠1；outer Active0/Total40不能视为40个唯一OS进程，breakaway叶进程缺直接Job成员证明。result e279b678…/root6c16570c…，2671前后列表相同但原sources_unchanged=false保留；旧入口禁止重启/放宽23门。独立有限证据report e515a52e…，两case成员诊断静态在制、尚未执行，生产driver/GPU/ABBA仍未启动。

> 最新§8.65：原根legacy两项首次CPU实际PASS/exit0/d649a0，2625来源稳定；六目录同session45554/PID60672实际exit0/24997d，6/6成功、19447来源稳定，result a1ca3bff964385f553ac66e3ff56757e7330a3b103ef7557d332a886b6718011，root bc61af3f3eb288989f32984d773c1b329d3f00c9c30ae697120de72ec8983e6d。driver r2十项首次CPU实际PASS/exit0/dd0021，2641来源稳定、334→416imports，result2ce179a2…/root593e12a9…。v4控制r2独审通过并首次启动session71940/PID7892，plan5bddc128…；尚无终态，只续看原session，禁止重启。新selection GPU/ABBA0；外层整树Job控制与实际超时清理仍待完成，旧control v1不接受driver delivery v2，须新控制接线。三次旧catalog失败原件保留。

> 最新§8.64：r4预算/路由两CPU检查实际PASS/exit0/b40191，2611来源稳定，resultbb8c0721…/root1769018a…。六目录首次session20922/PID47904实际exit1/a63ca9，首例越过预算与环境、一次Protocol后拒绝旧配置相对计划路径；19427来源稳定，0目录/ledger/GPU/ABBA。resultcf8880540cd130a2b0179a3f42324231c7b9bfb9fd3bcef12dd4864f1b61e475，root68eb80f7089287f612b82ff5261c2229a165e0610c90c6b9c3deb703fb1b4bcd。原B11 cfg/plan SHA均未变，确定是snapshot legacy模块ROOT误解析原cfg相对plans路径；须显式原根模块Source接线，保留actual96其它ROOT/proto。三次失败均终态，禁止重启。driver入口已静态封存独审，4检查未运行；v4/new selection/ABBA未执行。

> 最新§8.63：r3环境/首错2项首次CPU实际PASS/exit0/60de86，2608来源稳定，result0710fcec…/root58c93f1b…。新六步plan88ef5c5b…实际session32297 exit1/bf9170：已越过环境/341imports，但原control workload性能项11超旧validator上限8；resultc14e97b7ef5c66031aa3f026f935544281ee4044e33127bc3a7f7321449c7e45，roota5d562ae8905033031904b0df4c6c5e71225b1e8e5581c4d0111668447d79e4a，19404来源稳定/0目录发布/Protocol/GPU/ledger。两失败control及空父目录保留不重启。r4拟分开合法旧control B11(4,5)/B15(5,8)与原restored(6,11)工作量，不改validator或完整矩阵；源派生中。v4 CPU控制包静态封存独审ae435689…/578d8234…，未实际freeze/history/ledger。driver独立_child的parent传递仍待接线。

> 最新§8.62：六catalog r2的4项首次CPU已PASS（session58828 exit0/b010b5，2607来源稳定）。实际计划5e36f23f…首次运行session54368 exit1/b74869，首例在catalog/Protocol前因Windows环境key大小写检查拒绝，result31ec255e…/root861ff35fbea808a40790339076e179a2f92e94d40627a2ef88f354b66b6e7a08；19389来源稳定、0catalog/Protocol/GPU/ledger，原control不重启。有限同venv环境probe实际exit0/cfa9d7证实75键值一致，仅19原键大小写不同（observation6c8cb0b5…/roote08b18ad…）；r3最小修复与新检查待完成。v4入口3首次CPU已PASS/session94830 exit0/d187f8，2606来源稳定，result96e7d080…/root801b5fca4c0aa63dd28a450b3a8279a42ed873b60c87c4dadd379ec8a93cf5f4；非真实history/ledger。16采集登记已成功，后续实际catalog/v4/selection/ABBA仍未完成。

> 最新§8.61：Windows Source路径修复首轮UNC测试预期失败原件保留，修正后7项实际PASS/exit0/chunkec9e60，21来源稳定；result436a03a5…/root88d26487…。freeze-only独审通过，22保留函数与原组装器文本一致，旧Prepared生产者不改；同一session37895 actual exit0/chunkfcb204，16份v3采集登记已生成、19106来源稳定。target/unified-quant-restored-selection-mixed-registrations-r1/freeze-result.json SHA3409e2cb550d06bdfb8fd65f3392ab4203b7289e1f41840307748ef6ed9ae2aa，根退出e7c1ba921427dcde67ca17df5d5fbdf89e0089e66a79294cdc56f17b4059ebe7。GPU/新账本/预约/ABBA均0，原16预检与模型验证不重跑。后续六步CPU catalog r1未执行，独审发现venv redirector PID错误等同；r2修复及4项首次CPU检查待完成。catalog/v4/selection/性能/holdout未执行，旧消费与负结果保留。

> 最新§8.60：selection同一session20024 actual exit0/chunkf30e77，一次prepare＋16原生CPU preflight全部PASS/311.5s；20271控制/12298核心来源稳定，控制result d65f85f0…、Prepared9d7f0353…、CPU-results e2ebd555…、根退出da7b515c1236cb38c008f28613dd48b7ec9effc5bee48cae0d9fdbe4c5f428b7。随后freeze直接-I的sibling导入失败原件保留；改-B后session94549 actual exit1/chunk15dcc1，parent Source路径比较拒绝，body失败回执c31cf0bff9c310dcb031925b22055844826e6989f9a43ffb48d45e366e056a40。mixed-registrations-r1不存在/GPU0/ledger0；原生receipt extended DOS路径的Python等价比较修复与freeze-only新消费者待完成，禁止改旧原件或重做16预检/GPU。B15 GPU8与独审、mixed36均已终态PASS；旧永久消费/负结果保留。生产typed profile补丁已静态封存独审（delivery efe362bd…、14项仅定义），未应用/构建/测试/部署。ROOT-LIVE-STATE为当前入口。

> 最新§8.59：B15 同一 session35450 actual exit0/chunk42fe45，8/8全部保存恢复逐位通过，1969.468s；result SHA0a008391d531b34b9de27793447f922fe0ff2236f8596c1f242b62f4d769ded9，根退出556bcf6d…。mixed36首轮全部PASS/exit0/chunk30511f，2589来源稳定，root56d59610…。B15独审r2首轮PASS/exit0/session55881/chunk90d0ad，5972来源稳定，440对raw/858144配对f32、88对decoded、8对包一致，报告a3439302ec85fdd8b324c22d5b9aa6693544371359cfe223e247006f1389fcba；旧B11精确1056输出委托不重读。17-child selection CPU计划已实际冻结（337ba9b9569dd085c7286f6227efe9957fd79bf776dd01de24bbde2751745f74），首次启动session20024/PID68584，终态待观察勿重启；此阶段GPU0/ABBA0。历史失败/永久消费保留，selection精度、性能及holdout尚未通过。

> 最新§8.58：同一B15 session35450/controller3196已实际完成4/8，C3参考及3混合候选四阶段/保存恢复逐位全部通过，第四组owned wait0/220.6415s；progress-04 SHAe92fbc3cd834958715e67eb329b500c8e3cbc2c0756315bcdb7e4587b64fbd15。当前b15-c8-mix-000，总退出未观察，禁止重启/窗口内测试构建。完整mixed selection assembler源码已封（delivery10dd7bcb…），Prepared/CPU v2逐case环境接真实旧8B11+新8B15链；B15独立audit r2已封并窄审（delivery0b9cf688…），新raw实际核验/旧B11精确1056输出委托不重读。36首次CPU定义/freezer已封（delivery2b305ddc…，reviewee557e9b…）；8源AST、freeze、36测试及17真实CPU预检/独立audit均尚未执行，mixed-tests-cpu-r1不存在。最终selection/性能/holdout仍未通过；新17child plan冻结在静态准备。

> §8.57最新在途：同一session35450/controller3196已实际完成1/8组B15，首组b15-c3-mix-000 supervisor57152 owned wait0/214.556s，四阶段actual exit0；native verify逐位一致，stage SHA c982c49805d7ef84151af72e0a69322282032b33d7a417171d16b1c1bb2ccc58。progress-01 SHA9b34e4dd82f2e40152f50e28817015b83d953b4fd9fe7fa0f5705036aa168cd5；已进入b15-c3-mix-001，整体未终态，不重启。最新ROOT-LIVE-STATE的b15_dual0_continuation单列新进度，旧失败仍保留。mixed selection CPU runner17child源码已封（delivery c9559ea5…），6项检查仅定义，窗口内仅静态工作。

> 最新§8.57：CPU派生r1因新目录被resolve(strict=True)误拒实际exit1，未创建输出/GPU0，失败原件保留。隔离r2仅修fresh路径检查并绑定旧失败，3项路径CPU+syntax PASS；实际session50660 exit0/chunk8cd8e9，3771控制/5024产物及引用来源核验通过，result SHA0f4ad517bc4a4ea30d8bdd0d7d9e43eaad22661fc28ff41519d6353d745cbb17。实际八组index SHAb4085991960021d414596dc2d1495b717876331b17916b51e01f59456585497a；GPU plan SHA5530d8a8be88330806d0ba2bef9e64abca1409df4fa162a786aa56f98039c267。B15新8组已首次启动session35450/controller PID3196，首组supervisor57152；尚无总终态，只续看同一session，禁止重启/重复已消费尝试，窗口内不测试构建。新16上传/264执行/1008行/probe0/ABBA0/14400秒，条件累计33上传包含旧失败。后续selection逐case环境assembler/17child CPUrunner正在静态接线，未测试/执行；精度性能与holdout未完成。

> 最新§8.56：新B15接续派生器/薄八组launcher源码独审通过；4文件AST实际PASS，首次6+8=14项CPU实际exit0/chunk6e215a，producer22444/ownedvenv47248 wait0，14PASS/0skip/forbidden、333→336 imports、2546来源稳定。result SHA5f0e7565ececdaca8e4d0d007400ed19d972cff59381fd8fe1614b8b34da3a89，root退出SHA7108a5894f0f6609fd006927a2077f49ea343b9cc92243101a8faeb5784382c9；target/unified-quant-b15-continuation-tests-cpu-r1已终态，不重跑。条件累计load/upload33明确含原失败各1，执行528/行2016不增、Dual16；控制继承1+1703秒，新总限14400。完整derive未执行，新GPU登记/index/launcher plan尚未生成或启动。selection逐case环境helper静态交付及10项定义未应用/运行；仍需真实B15 record/verify与后续选型性能。

> 最新§8.55：B15 DUAL0一次prepare和八组原exe CPU preflight已首次实际全部PASS，session91222 actual exit0/chunk351452，outer81588/controller78760，owned controller wait0/64.7591928s；3277控制来源前后稳定。plan SHA3a637a7666120d4312726d2113e44edae73c3e472b76da9973abec4b6f544196，CPU结果SHAa9c2bfc35848d09533468b94819836ad8eca913f15358934f1496b5e6122a57d，root真实退出SHA b37359d16201a5b3049d7926b186ad8ee3617575ff8fa9e3f2b713345aefd760。入口target/unified-quant-b15-compatibility-cpu-r1已终态，不重跑；环境仅DUAL1→0，原模型/recipe/exe保持。这不是GPU/数值/性能通过；B11八份成功保留，B15失败load/upload与时间仍须完整继承到新接续，尚无新GPU登记或启动。

> 最新§8.54：B15 DUAL0兼容prepare及partial auditor r3已封存并独审；5文件AST实际PASS。首次16项有限CPU检查（4原审计helper+4部分终态+8兼容）actual exit0/chunk5985a2，producer17504/owned venv51812实际wait0，16PASS/0skip、335→336 imports、2524来源稳定；result SHAc8b41d340f980203d9e7aace79930dc7b3f86ddd0ae5fd0299e9abcd694db1a4。随后首次真实partial审计actual exit0/chunk550b42，PASS_PARTIAL_TERMINAL_EVIDENCE，4898来源前后相同，440raw pairs/858144 paired f32、88decoded pairs、8包逐位一致；result SHA0792400f9883c0e23c4bb2cd452d097c920cfcd4d8fde78a4866f35bc81e91a4，外部退出SHA1fed72022b7c49ae2689d9e7623b9b142b271ca0d448cc37b92c871fb5b2ed03。确认8成功/1已消费失败/7未启动，不是16成功或性能认证。B11不重做；B15新prepare及8个原exe CPU preflight仍未执行，9-child runner/新环境冻结在静态准备；尚无新GPU登记/上传/额度。

> 最新实际终态：同一session69973 actual exit1/chunk6fec6b，controller PID40860已结束。8份B11全部成功；第9组b15-c3-mix-000 seal actual0、record PID40488 actual1，后7份未启动。错误为quantization recipe has no eligible FP16 projection for DUALFFN；旧assembler把K512的B15也统一设DUALFFN=1。actual386只支持FP16 K384/H1152 DualFFN，现exe已有DUAL0合法Projection路径与probe=0，无需放宽形状或重编。失败record已预约model_load/model_upload各1，不能当零消耗或重启；永久seal/失败原件保留。总result SHA69a5a18111de1483fcd2dd9f54ac18f2f7250d7c3f09611464f4ae77399e6715，真实outer exit SHAaa53534db791b5a833552e0f3a693ec85e76b1f25aee5bac74ede240ea0f2dcf。ROOT-LIVE-STATE已改终态。后续须保留8份B11，独审成功/失败消费，再派生B15合法配置的新登记；原16-all-success assembler/freezer前置不成立，禁止直接运行或重做B11。见§8.53末尾。

> §8.52在途更新：同一session69973/PID40860已观察8/16完整通过（chunk cc1fbd），B11 capacity3和8的参考及各三候选全部record/restore逐位一致，8个supervisor owned wait均exit0、组来源稳定。当前第9组b15-c3-mix-000，总终态仍未观察；progress-08-observation.json SHA6341af4783fdc1ecb2b405f8695a8d2d15353a5fcc0da7c3143ca777e3f4bbed。最新入口target/unified-quant-candidate-campaign-launcher-r2/run/ROOT-LIVE-STATE.json；只续看同一session，勿重启或重跑已消费组。后续selection assembler/CPU runner及16包独立审计器已静态完成并窄审，10/5测试仅定义；窗口内不执行。见§8.53。

> 最新§8.52：13项续接/launcher CPU检查与一次当前显式环境38/38观察实际PASS；首例合法一对一续接已实际生成，index SHA6b43ca1d1c6d8d190a69dd8c4391ce33cd2179fb88d5726fd1ffb2409936da93。r1 CPU freeze的参数拼接及Windows路径拼写误拒原件保留；r2仅修Source内容核验/保留原路径，10项相关CPU检查PASS。r2启动计划SHA9f2b4479d9b235aa1de0d1bcac6b792fe2abc611c2a66ac3ca81abe87550e464，session37178 actual exit0。16串行GPU包验证session69973/controller PID40860已观察1/16完整通过：b11-c3-mix-000四阶段actual exit0，verify-stage为VERIFY_STAGE_COMPLETE_BITWISE_EQUAL，SHA977611c054f99f4a8febcc7dc3e3e5be59661817ef6cd5367cf75eaad4bb4ea8；当前已进第2组b11-c3-mix-001，总终态尚未观察。入口target/unified-quant-candidate-campaign-launcher-r2，必须续看同一session，禁止重启/重跑原永久attempt；预算仍32上传/528模型执行/2016物理行，总控制28799秒。GPU窗口仅静态并行，selection/性能/holdout未通过。原正式binary、生产Worker和旧10/16负结果保持。

> 最新续接见§8.51：同exe C32临时RPC的admission/runtime/显式offline数值与ABBA reader已隔离实现，实际19+34=53项CPU mock检查全部通过；admission direct exit0/chunk355504/result169e23e5317aba7dd9698ebcae9989a501ebe0eddd1de2ab1ade69a7c52261ea，reader-cpu-r2 direct exit0/chunkfb59cc/result70553b907c74714440053be33dcb339975e5367d78de87abf9bfe2bcb019de1a，2493来源稳定。r1测试语法导入失败原件保留，r2只少一个括号，业务源码不变。随后首个冻结campaign b11-c3-mix-000唯一启动direct actual exit1/chunk25808b，supervisor environment differs；attempt为CONSUMED_NO_RETRY，result phases=[]/seal=null/GPU root不存在，0seal/native child/上传/GPU。外部退出观察SHA4ce540f0c0cbe8f32682e027513f5cd108bb166855cbf3d65f7558685bc6b3c6。不得重启原入口；后15组未启动。只读诊断f53feac5a683621b2f143ebfe25c5bf7ff29bd2e644970a6427a2ad706859fca指向已实证的显式Popen(env=values)/reviewed Attempt，不能再用PS父环境变更后&继承；本次实际env差集UNKNOWN。首例须合法一对一pre-seal控制续接绑定0phase/0GPU失败，不能偷换目录重置。GPU包、selection、性能仍未完成；原v4ABBA0、旧10/16负结果、正式binary和生产Worker不变。

> 两条CPU链已完整终态：caller-cpu-r3/session81427 actual exit0/chunk51d2f8，仅剩余5项PASS（unit429.142s），imports438→448、7481来源稳定，result SHA4c6d5297b1df1175738e950b3cc2e92c4ad89e5d39fd9f822a35689a5c6d82b5；此前24方法局部ok继承，r1/r2聚合FAIL不改写。candidate-package-cpu-r2/session19770 actual exit0/chunk3b0806，10 synthetic+prepare+16真实CPU preflight均exit0，共18child；3188控制/3194输入来源稳定，result SHA79ebd2dcc9478764cb6233bac5f0eccb77d18bd3804433fae041aab09223cec7，CPU结果SHA3d06dea47e134cb077a79086ce3529b2a2d37b27722c8eaba2eb3da149d3cf26。原r1未绑定helper缓存的0child失败保留。freeze/session99734 actual exit0/chunk7941d5，target/unified-quant-candidate-campaigns-r1生成16未seal草案，result SHA607d24543ff6592e0ce434a2660594bdac16185f550447dde32480d2a4619e9d。所有原入口不重启；0GPU/上传/真实账本预约，真实执行包、B3/B8 selection和性能仍未完成。同exe C32临时RPC路线已静态核对（handoff SHA6e120c59a30f4c29b7460b591ba6f9ae419f5e261cbb3f4bc5f650c0a3117a2e）：保留nnMaxBatchSize3/8、capacity32并发，尚需Python预Welcome单Worker门、包runtime键和offline数值/实际RPC资格适配；未实现/实跑，当前ABBA额度仍0。不能把预算草案当运行。详见§8.50。

> 前轮caller CPU补检r2已终态：session2708 actual exit1/chunk189239，前12旧方法ok，第13完整合成catalog在300s上限处超时，后4未执行，result SHA90d5d21163a182e39ec7ab8335a7f73fbb0bdaffa95af659c04089dc867c24aa。ownedPID35908 actual_exit=null/cleanup_exit1，实际basePID32524后来精确命令查询未见存活但OS exit未观察；4795绑定来源末核稳定不代表unit导入末核完成。聚合FAIL保留、不得重启入口；只读诊断剩余5项，0真实history/GPU。详见§8.50。

> 2026-09-27 最新续接见规格§8.50：两模型各2004个selection局面CPU导出session90657 actual exit0，result SHA f8404009991f594b1195e4d47a8cdb5e1d9f7c3582e388f73d026ac3653d703c；独立输入审计session7170 actual exit0，18,688绑定稳定、4,008 PB重建及8,016 f32LE文件核验通过，report SHA9227aa3d613d758ff7fa27cf33d17e8ec17365e08a89b55ac074c76a579d722e。只证明B1存储输入，不是B3/B8推理。外层catalog/continuation/launcher隔离集成累计177唯一Python基础检查通过；r5真实gate/EOF通过、timeout清理UNKNOWN失败保留。r6修复原deadline内drain后9相关检查及唯一timeout实际exit0（chunk38dfa0，无异步session），helperPID12892 cleanup_exit1/outcome timed_out；2673来源稳定，result SHAfa1352eb2981c11f0e95d0fb39aee8ce7313b8d6572fdc7b0d4149e790bc3fa8。该r6快照共87文件、含continuation-r2；后续r4显式receipt根及89文件快照已完成20mock和唯一只读replay，实际终态见下文。r3未运行，长driver调用方另在r5验证。
> Rust父证明14项及CPU test-link实际exit0，result SHA81790528993b04423d39666d5c1303d3c723389c426525ea9b388575d04c41c0。最终CUDA CLI首次Cargo exit0/199.53s，target/unified-quant-restored-selection-cli-native-r1/run/build/release/katago-rs.exe SHA dd665d03fe31739712a205eaa6f63852dbc27fee339dfe91f38067237e57bfdc；外层误要求embedded全部386源导致actual exit1原件保留，未重编。独立native-readback-r2 actual exit0，4514来源稳定，embedded384 workspace仅排除未启用的2个TRT shim，provenance SHA72bb2282d427a7d9290b1efc15b99be5c0807605a644de823aa129cfd83dbb2f，result SHAf1cdaf64cb780482b8e1201dedb533034a604166889f50281591e14b14bdd8df。未运行新CLI/GPU、未真实catalog/continuation登记；新exe包不能复用旧exe包。生产Worker与正式binary未动，旧10 COLLECTION/16 ABBA及负结果永久继承；完整矩阵、独立确认和holdout未完成。
> 历史适配r4已实际验证：target/unified-quant-continuation-history-cpu-r1/session67562 actual exit0（chunk fce01f），20mock与唯一只读历史重放全部PASS，result SHA5974daa31c7b5cee37bdaa05acc57ac446cd9d182c3290c4bf032249d9416f83。旧81,693来源及旧目录稳定，最终84,554绑定来源末核稳定，78新增scratch保存在独立receipt根；66事件、10 COLLECTION/16 ABBA及旧head未变，旧orchestrator仍NOT_OBSERVED_UNCHANGED，不能补旧exit0。禁止重启。长driver调用方r5已静态封存delivery SHA8fe3fb5e1ba5183a699e8d669689d3838fdfe16f0f537b1bf952663c54108321；caller-cpu-r1首次session39675 actual exit1/chunk c554dc，12新mock逐项ok后旧fixture缺snapshot/target，末核又发现生成protobuf模块未绑定，聚合FAIL保留，result SHAae9839a5312839d03c95cbb119684365b25ad860ab14ae987b6914eabf1d6072；仅CPU检查环境待窄修，0真实history重复/GPU/账本修改。
> 16同版本包静态预算见target/unified-quant-candidate-package-wiring-static-r1/handoff.md，SHA2ebbe43fe9ba90e4a648897348bdf717d33321466a0b8d5a9692d23df53c95d3；48上传/7904模型执行/34200物理行和Dual探针48仅拟定成功路径、未登记/执行。构造器target/unified-quant-candidate-package-assembler-static-r1/delivery-r2.json SHA4ad76afbce05744a57600cf6611d2617906f04abd04656d58466433dd59c5166已静态封存；独立50来源审查稳定，修复cudaQuantPlan与Windows路径别名，review SHAe4440bf45b3b325b41484a69d0307187cab1af2507bf84f308665bfe6a38688a。10检查仅定义，prepare/16 CPU preflight/freeze尚未执行，真实GPU包和catalog仍未创建。

> 2026-09-27 上一阶段§8.49：固定包 selection 接口及 DualFFN 17kind 隔离集成已完成有限 CPU 验证。Rust r3 session39906 actual exit0/chunk c9b5cd，421/421 CPU＋四次 test-link 全通过；result SHA bee19b8c123e7883fdbb20722de4750b33f0404897530a3732ff0e4361597f62，386 编译源前后清单 SHA 377adf7485e5c91c3ecab99614e555cc94a0503f89f8173a9e9d9a9b8c077fe3。最终5477绑定稳定，初5393为其子集，新增84产物。后续源基线 target/unified-quant-restore-selection-source-r1/cpu-r3/run/snapshot；初始source保留旧测试断言，不直接复用。Python r3 32项 PASS/actual0，result SHA6bea90ec6bd52c63618b2179752d54d3eec87dcbb22474961ea724fab9c3032b，2153前后来源相同、2155含日志；独立只读复核通过。各r1/r2驱动/fixture失败原件保留。CollectionPermit/collect/preflight/父证明/共享WorkerPB适配已编译检查，但实际外层启动器、新合法continuation、最终应用exe及同版本包当时尚未实现/构建/登记；无新GPU或生产变更，旧10 COLLECTION/16 ABBA与负结果继续继承。20/44仅建议，不是新预算；完整矩阵、确认、holdout未完成。见规格§8.49及target/unified-quant-restore-selection-source-r1/continuation-boundaries.md。

> 2026-09-16：本轮 Fork 对标性能优化已按用户要求结项，停止追加测试和实验。
> 当前交付与使用入口：`docs/RustGo性能优化结项记录.md`、`docs/RustGo使用指南.md`。
> 主规划和实验报告中的历史“active/下一步”不构成继续执行授权；后续优化需新的明确任务。下列开发与数值纪律继续适用于后续任务。

> 2026-09-24：用户确定“兼容已有模型的统一推理优化量化后端”新目标，目标规格见
> `docs/RustGo统一推理优化量化后端.md`。G1 逐投影 FP16/INT8 与 G3 显式 v2 MXFP8 实验路径已实现；
> MXFP8 三模型 B1～B8 Graph/双缓冲错误隔离通过；r8 全 FFN MXFP8 对 INT8 的 C1/C32 诊断均负收益。
> r9 缩放清零融合默认关闭，算子/整网逐位门已过，仅 B11 C1 的局部收益确认；其它三负载未过收益门。
> 有限 v1 种子的离线执行驱动已过 13 项 CPU 测试及一次 B15 微型实机流程冒烟；入口见
> `docs/RustGo统一后端离线选型.md`。旧 FP16/INT8 显式执行 spec、前后指纹与实际日志绑定已实现，
> 81 项相关 CPU 回归及一次 B15 两路径数值接口验证通过；该验证无性能 ABBA、仅同一盘 32 个 calibration 局面。
> 执行方案目录已过 30 项 CPU 测试及独立复审，完整控制矩阵连接状态见下文。新增合法前置 AB 导入通过 11 项测试；
> `target/ptq-pro-corpus-r2/corpus/manifest.json` 为独立复核的语料组成 READY：32/32/128 盘、2044/2004/8049 局面，
> seed 固定，旧两盘诊断不进入新 selection/holdout；包含初始布子局面，未验证搜索叶分布。详情见规格 §8.14。
> catalog-v2 已接通原始采集/完整 ABBA 入账、完整控制矩阵和外部持久账本 driver；本轮 151 项相关 CPU 测试通过，见 §8.18。
> 新探针与性能窗口统一 perf_counter_ns；旧无标记探针仅兼容数值证据，见 §8.17。五份历史采集的 test registry 不作未来实验预算。
> 首轮真实 selection C1 负载已预登记，32 盘/2004 局面、B11 后 B15、共享 9 次采集/13 组 ABBA；B11 已完成 4 份数值/5 组 ABBA，均稳定，但无统一候选胜过完整旧控制矩阵。
> B11 独立审计通过，16,477 个绑定文件前后 SHA 一致；保留原方案。B15 五份 2004 局面数值比较及完整 8 组 ABBA 已终态并完成独立审计；相对统一 FP16，旧 FP16 最大胜率差 0.219029 个百分点（非逐位等价），旧 INT8 q64 为 1.836920、统一全 FFN INT8 为 2.592498、统一宽 FFN ≥384 INT8 为 1.530784 个百分点。宽 FFN 配置实际 12 层 INT8/33 层 FP16，不据数值结果推断性能。
> q64 工具扩展 r3/ledger-r2 的 41 项 CPU 检查、编排器 r2 的 33 项检查已通过；B11/B15 首轮及独立审计均已完成。实际续接位于 target/unified-quant-q64-continuation-real-r1：新增 B11 全 FFN INT8 q64 的 1 次采集/3 组 ABBA 已终态，继承全部旧消耗与负结果。2026-09-27 核实 session85010 句柄已不存在、无实验进程；原始进程退出码未观察到，不能补写 exit0。prepared/result.json SHA 7a1d324fd72fdecd352b85c13afe388d835eba4cd8ab3beeee9e3eaf073a5eda。独立 NEWaudit-r3 通过，81,693 个绑定文件前后 SHA 一致；相对旧 INT8 q64 仅 +0.205606%，完整矩阵无正候选。r2 审计器 schema 假设失败原件保留。禁止重启、重复发送 gate 或清除根永久 seal。当前进度见 ROOT-LIVE-STATE.json 及真实 events；见规格 §8.20。
> B15 已完成 8/8 组 ABBA，三个候选均未通过 C1 完整矩阵：宽 FFN ≥384 INT8 虽胜过旧 INT8 q64/统一 FP16，但比旧 FP16 速度比仅提高 0.4570%，未达 1% 门。session59237 已 exit0；独立审计首轮 PASS、20,577 份绑定文件前后 SHA 一致，报告 SHA ae35ee6411c59115a2b0ccfbe4af9f4a3fb0ea7e3470a3a780aea45d9a88d76a。不得重跑负结果或据局部正收益进入确认/采用。
> 确认器 r3 的单次长等待已由真实 CPU 检查证实取消延迟 6.766s；新隔离 r4 改为≤0.25s等待片和单一总deadline，44项单测与cold-r4 7项真实检查首轮全部PASS，在途取消0.266s。r2 PID假设失败/r3取消失败原件均保留，base fixture仍不覆盖venv redirector在途取消；没有真实确认receipt或holdout。Rust执行计划原型21项、单FFN提案14项CPU测试通过，实际B11/B15提案34/46份已生成但校准敏感度/成本仍UNKNOWN。详见规格§8.21～§8.22。
> 2026-09-27：完整 FFN 成本接口与 ffn_group_cost example 已接入实际源码，17 项运行时 CPU 测试、5 项 CLI CPU 测试及 1 项实机 owner/context/input 拒绝测试通过。B11/B15/ONNX 的 8 组有限合成诊断覆盖 B1/B3/B8，实际融合/剪枝/混合精度路径及五头观察前中后逐位一致全部通过；report SHA e0eb29e783f5dd0e31963ed24544e63a295857b1bc0fd7e69d364da433a6ae7d。独立构建 target/unified-quant-group-cost-build，不是校准表、Graph/端到端性能或旧 binary 等价证据；见规格§8.23。
> INT8 LT 缓存持久化 r2 已接入实际源码；隔离 24 CPU/native-check 通过，实际 CUDA crate 首编发现 DeviceFingerprint 不支持 Serialize，已用 actual-only 显式四字段共用编码修正，实际 CUDA release 编译及 24 CPU 测试通过。随后有限合成 native probe 首轮实机PASS：2 stream × B1/B3 ×3逻辑形状，67次调用尝试、2次Graph重放、48份原始u16输出逐位一致，record/restore表一致；result SHA 02f740c0b7c91a606d7ff6c1ce78321739c827c9ce6a9fd4f7052e745ab0b643。独立CPU审计PASS，149份绑定文件前后SHA一致、48份raw共16,265,216个half核对通过，report SHA e67822a7bfdf3ec4f5be618c6ae0459548ebecb95c81ab1d39ec8bcba079b003；22份来源快照已独立核实。模型薄适配已actual接入并首次CUDA release编译通过（session6717 exit0/21.70s，记录在target/unified-quant-algorithm-model-api-r1）；随后实际B11/B15全FFN INT8、B1/B3的16次前向/4次Graph重放首轮PASS；80份五头原始输出逐位相同、四组算法表分别字节一致。独立审计222原件前后不变，report SHA d26bc535c01dcd3b0c41511410faa075ea88ebb6d7fba2a222abdb5a47f29519，53份来源快照已保存。target/unified-quant-algorithm-build 与旧诊断 executable 分离；仅partial INT8 cache，不是完整模型plan，正常配置不自动启用。见规格§8.24。
> 校准CPU输入编码器已actual接入，26项CPU检查通过；两模型真实parser确认v17/v7/22/19/meta0。首次prefix9主线程栈溢出原件保留，仅example入口改256MiB专用线程后，r2两模型prefix9真实编码与独立审计PASS（191绑定文件；报告SHA ff253d9f7e08ae8ce34a98cb10e773de72600328dc1ac7adfaa2c77530852cb8）。target/unified-quant-calibration-encoding-real-r2/complete2044 两模型全部2044局面CPU导出及独立审计已PASS；session13195 exit0，20,245绑定文件前后相同，11,928 f32LE原件/390,534,816字节，B1/B3/B8每packing全覆盖、尾B1/B4无padding；审计SHA 1e8af90f2329cc13d2013fab47b84323389a1aa0aa0630b52a8242469b657e80。当前encoder exe SHA946d3640c4d5f21ee367f26275d6601c142c74c650430294bced100445856017。仅输入来源证据，不是敏感度/成本/精度通过；见规格§8.25。
> 共享原CUDA五头解码/evaluator后处理/Worker PB接口已actual接入；74项实际CPU回归、真实B11/B15模型契约及6行合成头/32拒绝检查通过，26份相关源快照保留，见§8.26。后续固定输出回归已实现：旧/新隔离CPU与CUDA构建通过，4例合成CPU回放及4项新reader测试通过。GPU首个B11 direct两次前向正常exit0，外层reader误判Windows extended DOS路径导致FAIL，原件完整保留；r2只修路径规范化并过7项CPU检查，只续接未消费五例，共12前向/24行、20份完整五头/20,432个f32，raw/trace/decoded全相同，GPU聚合SHA af07dabad1c16cbc72fd10e6f8bec4a608d3adc628614a22a012419fd77a0326。旧Worker/新Worker/共享接口的六个真实输出CPU进程均exit0，两模型各4行最终PB和f32/f64 typed-bits三路全相同；1,054输入绑定及54新产物稳定，CPU结果SHA a08da6ee712e2669054183707452ac99fdc90d82c017c6013c1a266ef0828115。独立终审PASS/exit0，4,489绑定文件前后SHA一致，报告SHA b0c9d455476e27f9e61a82b63ad2fd411dc4d7ce6eae96f66767fc8156dea091；没有重采旧D或性能/精度认证，正式binary与生产Worker不变。完整校准敏感度/成本仍UNKNOWN，下一阶段静态交接不构成新GPU预算登记，见§8.27。
> 单配方校准collector已actual接入，24项CPU检查、两模型完整输入CPU preflight、80份配方实际parse/lower/resolve与14负例、CUDA release构建通过。新有限4case实际104次前向/404物理行、4次上传均exit0，结果SHA5a5d6613a4a81fb10b8fe8a519af32ab5a4fcf3b9b99740bda7aad25686bb66b；每例仅17个placement/12唯一请求，成本仅每B一输入W3/M2，完整五头测量前中后逐位门通过。指标适配器最终17CPU检查通过，两模型真实比较exit0，结果SHA8ae420e8aa8761965d39ac8aa597ca831f09234f916919978f088d68ebc7f764；均NOT_FULL_CALIBRATION，无精度/性能认证。独立终审PASS/exit0，3,843个独立去重绑定文件前后相同，173,672个raw f32全finite，60个成本样本/2,340个完整组记录通过，报告SHA e62ea1754166eb21462d958c54cc4ff1d9ff3145373bf9a63927ee5bc050d3b3；首轮审计器误判B15 padded FP16路由的失败原件保留，仅修CPU审计器，无GPU重跑。完整2044×80采集与选择仍未执行，外层driver、预算冻结和成本汇总待实现；见§8.28。
> 完整单FFN采集驱动、输入索引和成本汇总已接通，17+9+21=47项CPU检查通过，独立静态复审无阻断。全80配方计划及19,963来源CPU复核通过；target/unified-quant-calibration-full-r1/registration.json SHA a3f8e70649d1ca9cd8fd540d2c42fdb378efe0cd7215e4679f6b492a20112837，固定每配方2044个B1 numeric、成本B1/B3/B8各S8/W3/M4，全局175,760前向/212,480物理行/80上传。已首次启动session80213、driver PID44160、首例collector PID56880；终态待观察，勿重启。execution-consumed.json永久保留，消费与case终态以capture目录原件为准；运行期仅静态并行工作。完整指标、组合选择与收益未通过，见§8.29。
> 完整采集同一session80213/PID44160最新观察22/80份CAPTURE_VERIFIED，尚无终态，禁止重启。新增有限混合提名器scripts/plan_ffn_mixtures.py仅静态完成：固定18槽、完整recipe去重、无float admission独立SHA；19项测试未运行。多FFN Rust采集r2补丁与Python policy比较器暂存于target/unified-quant-mixture-{collector,comparator}-static-r1，13/17项测试仅定义，未应用/构建/GPU；Rust补丁SHA e214eda5332e7c03f8428a3f03fb6251be7c82391280bdf1d011724121cf0a9d。后续先完成原采集/78比较/提名，再在隔离checkout叠加当前dirty快照集成，保留历史Source原始绝对路径；新compiled_sources/exe参考不能直接复用旧捕获。底层支持多FFN混精度，但每handle配方仍固定，按物理batch精度调度未实现；见§8.30。
> 同一full80最新已观察B11全部34份CAPTURE_VERIFIED/exit0，PID44160仍live，B15待完成。新增target/unified-quant-mixture-driver-static-r1的准备/运行静态稿，15+16测试仅定义；唯一逻辑consume根继承父full80，最多20次新参考/多FFN上传与40,880 numeric B1调用，未登记/执行。history锚SHA 7cf8b6c2dd0934dea584d3490484055dc345b174563b09323bc351b62fe16f42保留旧10/16消耗和C1负结果；r3 collector增量记录CPU preflight真实backend_build，仅定义1测试。v1按§3加载时固定精度，动态batch精度bank不是前置；FP16 LT持久化及完整执行路由/生产接线仍待完成。见§8.31与各静态handoff。
> 同一full80后续观察48/80（B11全部34、B15至013）CAPTURE_VERIFIED/exit0，PID44160与session80213仍live，禁止重启/在窗口内跑测试构建。FP16 LT静态r1已封装于target/unified-quant-fp16-algorithm-static-r1，delivery SHA f77d0ff75576ea66a97d7614e8ee802fe7b429b982915fa37cc3a2d68e818e7c，9源/21测试仅定义、未应用；每stream独立32MiB，保留TF3preset和双流，自动模型目录仍待增量。新增多FFN CPU聚合入口/12测试仅定义，已静态修复父原件/运行库来源及CPU receipt链缺口；CPU checks/build/preflight编排器静态完成/17测试仅定义，绑定实际Python与Cargo产物，失败停止且Cargo后代状态UNKNOWN；皆不构成GPU登记或精度/性能通过。见§8.32。
> 同一full80最新定点观察62/80（B15至027）CAPTURE_VERIFIED/exit0，PID44160仍live、无driver终态。自动模型LT目录静态增量已封装，delivery SHA 4eec720809dafa256c975d7d0c22dfb4493aa9dbf115ded1aec2f019c746d75e，16测试仅定义；目录保留非LT路径，不是完整attention或逐op实际追踪。新增同stream共享FP16 scratch静态交付delivery SHA 9bda23476accedd8708e481e6f9b41c2875d2d7a01f9054a569f027a57ebc8a1，5源/9测试仅定义；新闭包提交入口在Graph之前检查并与Lt共锁，五头DtoH/status在handle单提交锁下分段上报。正常backend保留graph/recipe、完整batch覆盖、package profile、Graph生命周期及失败清理仍未接；详见§8.33。所有新静态源未应用/编译/测试/GPU，禁止窗口内执行或修改冻结原件。
> full80 已完整终态：80/80 CAPTURE_VERIFIED，session80213 actual exit0（chunk b7bce7），PID44160 不存在；capture/result SHA 1341821a66fb302ca61a33b339dd3e3a0df919ec28d438b4269041f619f99320，driver exit SHA b5c08991bfab123df90b9be1b093ca147414c49998bcf5fcf34294cd7f4b0f4c，sources_unchanged=true。原永久消费不删/不重跑，数值比较尚待实际CPU receipt。attention增量delivery SHA 063e37117420bcd002a440932373972585f538a35dde805f0a3758324c1f484c；组合执行包delivery SHA 5aa384dc548637d89d67c7ed065281bd070f94e7f171d922a89497fe679a2512，composition一次应用于隔离当前dirty快照，核完整batch/实际目录/叶表/owner与失败状态；13+23测试在封存时仅定义。正常backend/profile/Graph生命周期尚未接线，见§8.34。
> 原版本78份CPU比较已首次启动session22049/wrapper PID78888/比较driver PID35220，比较器8与提名器19测试均实际exit0；target/unified-quant-calibration-comparison-driver-r1/cpu-first及comparison-full-r1原件为准，勿重启入口。执行包隔离CPU r2实际106/106 PASS，947绑定来源稳定；result SHA7d4db90653eca6b950fe741054f66baaada451d79ef57ecfcc849cdd450bf992。r1依赖选择失败保留，r2只修runner、组合清单仍总计应用一次；未做CUDA/native/GPU，340文件CPU快照遗漏AOT参数.bin，后续native须新派生补实际输入。见§8.34。
> 多FFN采集源码已独立集成至target/unified-quant-mixture-integration-r1/r2/source（与执行包分离），499原源/513最终源，integration SHA4fba7dc5642112cfb97ebfeb1049ab18128fa4e002afbbf73aa53cd5980c8284；r1 CRLF严格核验失败保留，r2仅核换行后采用冻结staged精确字节。helper exit0、未测试/构建/plan/preflight/GPU；等真实78比较与proposal锚再续。见§8.34。
> 执行包首次实际CUDA release库编译exit0/110.187s，native-r1退出receipt SHA f457eb1a2cf14200f8d2642e95f9ef3a139333a40871ad949d4c760289725b3b；r1仅postcheck文件名误判FAIL保留，新只读native-readback-r2 PASS，result SHA754390af9447f8a101868b7cb977a8a595590667219b48459212fe9dab2fbdf5，5059来源稳定，0重编/GPU。正常backend接线草稿在target/unified-quant-backend-execution-static-r1，配置双键/固定包profile/全batch安装/Graph前置检查/失败清理，4测试仅定义，依赖共同actual Binding工厂，未应用构建；诊断example包不得给不同exe部署。原CPU比较最新45/78（B11全33+B15前12）均6pp PASS，仍等完整终态；见§8.35。
> 原78比较/提名已完整终态，比较/提名child均actual exit0，wrapper句柄session22049已不存在、不补写其OS退出码；result SHA9bb59fdc588ea847c2e3f05336001dda72e9610a82cf8271f4c7618231da7b1a。78份B1/2044校准均6pp PASS；12混合提名来自B3/B8，B1六槽空，proposals SHAfbb1e3a042e11d5c2c8e5491e86bd5ddf1bbb825d754ee8f69e6c9701eacfae5。隔离14case plan-set首次成功SHA6f28fa44ba8222368292cd2919efea5e334c01bbb22f8ad89fd9fac6b481f1f3；首轮CPU流程54PASS后因隔离target缺编码fixture在第5suite setUpClass失败，原件保留，未构建/GPU。精确16,088文件fixture复制后新输出目录重验原CPU流程，不重置GPU预算。执行包352文件最终集成r2首编E0277/exit101保留，0新测试/GPU；仅修build provenance参数错误类型的final-composition-r2 SHA49b263fae4e65c844176f81e48954ce8881015ea835e47e343e5db7b0ba6e3b3，integrated-checks-r3/session2929首次运行待终态。详见§8.36；不据静态稿或单层PASS宣称组合/性能认证。
> 同一integrated-checks-r3/session2929已actual exit0/PASS，143/143CPU（三次Cargo实际链接成功）、5,525来源前后不变；result SHA8ec148782cb7a5beef2499f04af0cf529b09f5a0cd03f6a245fd32e1f7c9760e，probe exe SHA92ca23d9eded73331c5bb5be7451b0dae2023a16bc70812862875d138769f6c8。实际embedded provenance非Null且所选静态cudart/源字节通过；未运行普通probe/模型/GPU/Worker，无需重跑143项。正常Backend运行验证及同katago-rs recorder未完成，方案见target/unified-quant-production-recorder-static-r1/handoff.md（SHAd1e8dbbe59ba81cc7a23c228f39a4f787c2c7488011b6a6b6f95ad319af3537c），不得将example包改SHA部署。
> 执行包六case实际CPU预检PASS/exit0，result SHA025e285e797dad41a71eade9e00733b233ee8edeb0148d7896d7d89168ae11cc，registration SHA3d1330bf5a72db8a325bc989d23855b37e1f354ff9c436469b65ee0d199482e3；5,553绑定来源不变、36合成输入、0GPU。已授权原注册一次有限GPU诊断（66host apply/12Graph/144rows/330raw），driver待launch/终态，不得重试或据此宣称精度/性能。混合CPU第二轮在第11suite发现fixture漏plan而停止；新r4隔离源仅修该测试，17项已PASS，现session33886/wrapper65796/preflight74936在r4/source/target/mixture-cpu-third跑原完整CPU/build/preflight，勿重启，不重新生成14case计划。详见§8.37。
> 后续r4原完整流程230CPU检查已PASS（168Python+24/38Rust，专门17项另计），同session33886在首次CUDA build，真实Python12244/Cargo67248；14case CPU preflight待终态。执行包首次GPU诊断已启动session42692/probe21156，driver target/unified-quant-execution-package-probe-gpu-driver-r1，严格原registration3d1330...预算/1200秒，不测速、不重试、不删gpu-execution-consumed.json；终态待核。生产Worker78408保留。见§8.37。
> r4混合CPU链session33886已actual exit0：230项/另17项修复检查、CUDA collector构建与14/14 CPU preflight均PASS，inner result SHA101086d3c72aaad6482ea5082c68c762c646079da47ce27393f33598c8500f6e，exe SHA0367870585a65902fe86246792779d356957fbd848a206076dae0e1bef34755c，3573来源不变，未混合GPU。执行包session42692亦actual exit0/PASS，driver result SHA9d43f75d4ad22b68dc31daf8f8a5f872c6a82c7950b5ee03dcfc9731e836e2a4；330raw/240全头逐位比较及六套四份package相等，5557来源不变；consume SHAafe536119c5b000a7074733e8cf57e1d10ba29946ea83f617de2c88ddff55eac永久保留。仅synthetic CudaModel API，Graph B3/两stream顺序，正常Backend/Worker/性能未证；独立CPU补审逐case计数/所有新产物/DLL中。见§8.37。
> 执行包独立CPU终审PASS/exit0，report SHA06d0f04b96d4eef81587285da74110abbd2a8de14a9c4c846dc491b0d858cdcc；732逐case/batch事件、330raw/240全头比较/24包、6107绑定文件（含全部547新产物）通过，未重跑GPU。14case混合注册CPU freeze exit0，registration target/unified-quant-calibration-full-r1/mixture-calibration-v1/registration.json SHA8bb08018e6e260090ff53205a662ae3d2cba792f006e5d2e62d0f27b99398e1e；保留旧10/16与负结果。已首次启动session75714/wrapper80208/driver60444/首collector82620，外置target/unified-quant-mixture-calibration-real-r1/run_gpu_once.py SHAd812c51f069dd12314303a38422b5dc3ef925477080dd818334d05903df4011b。固定14上传28616个B1numeric、cost=[]，25800秒总deadline含prelaunch；日志gpu-first，capture在原逻辑campaign。当前在途，勿重启/删除新consume/在途选型；终态后再12份CPU比较，Worker78408不动。见§8.38。
> 覆盖上一条混合采集在途状态：session75714已actual exit1，driver60444 exit1；首B11参考collector82620 exit0，已2044前向/1上传，producer SHA8fdade245705a10470fab44020e4e74ff0b8478a11dacb01d7853795e25df2ee。外层CPU descriptor_equal将Python manifest Source索引误作Rust manifest_path描述，KeyError后停止；capture结果SHA65cb83abaca2679163d51bcdb6e03d5552e6d3649ee45b5ae14df8fbb467e73f，reserved1/observed_forward_attempts2044/verified0，sources不变。80208/60444/82620已不存在，Worker78408live。首例和永久consume不能重采/删除，后13例未启动。正在target/unified-quant-mixture-descriptor-recovery-r1新CPU reader显式投影/负例与32chunks2044rows只读回收；不改冻结r4源/registration，不GPU，合法继承消费续接待完成。见§8.38。
> 首例新reader只读恢复session17257 actual exit0，16项CPU检查及2044行/32chunks核回PASS；result SHA793539e99f5612dc305ccd02607e5f0759d0513b21764ac50aa689fa67dab811。新reader SHA98b68e2444b7f4e2269f7c48f8aca497d186961a3f44f620b738e6468922532d，严格映射SHAd9cbf35ca3faff1a088e8cb01b0dae77863d9ef511e2ac2d873459581103cebf；最终10338绑定来源recheck稳定，before4086是after子集，不是两清单相同。原FAIL/consume/r4源不变，0GPU。剩余13例续接和新CPU聚合准入正在隔离准备，全球继承1/2044而非新空预算；更早硬截止2026-09-27T12:30:00Z，仅静态未启动。见§8.39。
> 续接20CPU检查与CPU freeze均实际exit0，registration target/unified-quant-mixture-continuation-r1/registration.json SHAe39844d14e8ef9c90e4e31652cc43cfe4d6e2f31e84f7686bdb2d3c3b8ec47b5，继承10357来源，driver SHA119c4d7bfc27329cd5541dcf42f9d82d22d9948772c62989abfc362bb1722fd1。根已唯一启动session38519/wrapper69984/driver4632/首新collector76368，日志target/unified-quant-mixture-continuation-launch-r1/gpu-first；原capture继续写新case，新aggregate仅campaign/continuation-r1，新continuation-consumed.json永久保留，勿重启。原B11参考复用，全球14/28616、新增13/26572，硬截止12:30Z。运行期只静态，不tests/build。比较适配r1路径失败原件保留、r2九测试PASS，r3加重复继承字段门后11测试PASS（SHA5991fbafde14d5450ae66d3c204bfbf5e9d419d136023b1295950408519cb576），未真实比较。见§8.39。
> 同一混合续接session38519已actual exit0（chunk be09af），14/14验收、全球14上传/28616前向，首参考复用、新13/26572；driver4632 exit0/1227.8193863s。capture SHA0870195ba290fc9b44c64d1e905150a77ad3fd0ace3cb5be0f882851534f5ff6，wrapper SHA4b90b36ef435da9358ccd82989e24cf8a205878831d6e5a84bebd79cffb3f868，tool观察SHAcca801ea921427c6d62c96652deaa6e47dd28b77b5774c1ffd9bebd4cac6797d；两永久consume及原失败不动，禁止重采。独立完整性审计/12混合数值比较待完成。新正常recording API静态r1经独立复审发现panic清理顺序缺陷，r2 guard修复delivery SHAa6376fb6ae1608060325d460861c221f00e4a94b6e30f0ace8680552d86c4ae5；共用setup/eval配置r2 delivery SHA9f955397c7c58d89215cde486c99301a5f7e939d04e6c0e688307221ea18dd33，独立静态复审无成功路径阻断，披露错误顺序/used-key差异。两增量尚未应用构建测试，14+14CPU仅定义；见§8.39～§8.40。
> 混合采集独立终审session78638 actual exit0/PASS，report SHAbc365b1988aedfbcfcb99ce5e3dfa76c46dbedf05f0fcf8a2f046429ac98ae01，22,363绑定Source/1527产物稳定、73,085,264raw f32有限。12份混合比较session56071 actual exit0，index SHAe99c674dd228e86135d92619dcfce3d1904900876f5939c3cba85bc885477ca4；各2044个B1局面均6pp PASS，B11/B15各候选最大胜率偏差最高2.406883/1.530261个百分点。只读P95补充session4447 exit0，report SHA27122475fa57b913b59010187e30958f587b6ecc7c0992cd45370c27e5a4fe24，P95最高0.420713/0.339818个百分点；原mean/max/KL/top1精确重现。成本提名来自B3/B8，本次实际数值只有B1；旧阈值12份仍FAIL，无selection/holdout/性能或采用，B1成本与历史C1负结果不变。见§8.41。
> 正常recording API+共用setup/eval resolver隔离集成首次session21051 actual exit0（chunk840915），106/106 CPU与四Cargo阶段均PASS；result SHA7860e4efe0b412b0166be5952f5954208456e26426d4651030c6a66eec183cba，实际356来源清单前后SHA971ab4879136b7be623277e0cef6a1be22239f6b5b71273d33d286656e7cc24a，4531绑定文件稳定。root/旧快照未改，尚无正常Backend双槽实机/完整CLI/生产发布；CLI静态增量须以target/unified-quant-production-recording-integration-r1/run/snapshot为基线，不重复原106项。见§8.40。
> 修正record/verify集成r2已完整终态：session98926 actual exit0（chunk96fa1f），226Rust CPU/五Cargo/三模型CPU preflight全部PASS；result SHA84256a913c2b80e3413b072249970416f2665af1284b86dba106d3c23f5cd67b。375源码前后清单SHAcd08be23ed03f7fbb9d59490d41148d64918ee7038544a632ed2489b7a8c9c25，4,691绑定稳定；继承21+2监督器检查未重跑。最终exe19,060,736字节/SHAa9fa63ae8e864d6a4793fa5fea220405538befebd306d9f7a3c553cce9fb4a9e，embedded provenance SHA074b4e6fe428d192c5aa16a9e6627eaf05197fe472c02761b70c9539192ffb80。0GPU，正常record/restore尚未登记/执行，正式binary与Worker未变；失败r1原件保留，见§8.45。
> 正常Backend三例有限record/restore已完成首次CPU准备exit0（chunk92a9ed），target/unified-quant-normal-backend-record-restore-r1/preparation-result.json SHA8fc4e5104ed367f7c79c0ef78529e1d8a2f0d245579d863367b38345f9be2b15。B11 FP16/B15首FFN INT8/ONNX FP16，六stage总54真实前向/108物理行/最多336attempt；每例1800s。原监督器、固定环境、首失败停止，不重用旧预算。CPU准备无seal/claim/GPU，待逐例唯一启动；见§8.46。
> 正常record/restore首例B11外层监督器actual exit1（chunk035a96），result SHA3af9b9aad33a22854f062545b2707e64cbb0c70cfd8122206e156a6f6690f7cc：环境不符、phases空/seal null，原attempt永久CONSUMED，禁止清除/重启。0Rustchild/0GPU，B15/ONNX未启动。两个有限CPU环境观察确认原native传递90/预期38键，显式Python env传递38键全MATCH；监督器不改。仅B11未执行阶段的显式替代登记待准备，原三例合并预算不增，见§8.46。
> 环境显式续接仅替代B11，replacement-index SHAa1e1063866e726c46382626b01cfb3adecacc044ada0a8953b52e961bd8ee9e2；原失败永久保留，GPU预算不增，三例余5399s。B11 session61351 actualexit0/c0536f、B15 session1359 actualexit0/bf7ca9，均四phase0、30对raw/30648f32+6对typed bits全等、re-export全等；B15真实INT8 marker确认。ONNX原登记唯一启动session34347（chunkb4e68f），终态待观察，不得重启；见§8.46。此处非性能/精度认证，正式Worker不变。
> 正常Backend三例现已全终态：ONNX session34347 actualexit0/ab153f，sup SHA79ad2fe6c126e6d572a8e0d50ebc69b2d1dcf5ef86244d5cdd83e0e0199357f2，verify SHA7d6919746b74216659c91d307fdc0d6ac50069a35c31b56a38abb7e128008468。三例四phase各0，90对raw/91944配对f32+18对decoded全等，三包re-export全等；六stage合计54前向/108物理行。GPU窗口已结束，不追加重放；独立CPU证据审计待首次执行。仅capacity3/三局面来源同包恢复一致性，不是精度/性能/Worker认证。DualFFN observed probe新增17th typed-kind方案仅只读设计未实现，见§8.46。
> 三例normal record/restore独立CPU审计首次actual exit0（chunk3ff7fe）/PASS，report SHA3ff1889bfffc5487e8000d0edcb43c849811ceec388af09128bc1b4fe6fab374，852绑定来源前后稳定；90 raw对/91944配对f32、18 decoded对、3包对及54前向/108物理行预算核验通过，0审计GPU。真实退出已保存，不重采。后续DualFFN真实probe hook及17kind版本/计数仅新target源码开发，尚未应用/构建/测试/GPU；本轮GPU窗口已结束，正式Worker不变，见§8.46。
> DualFFN真实probe及17kind预算静态增量已封存，observed delivery SHA55ef531757d02cbd43f8cbf9b0fd8211a50a15f8bd822fbe01cfea8235c17560、budget delivery SHA79525bda9eb6cff83301beba7b8d0bf2d840d9916ab91fa0259dd9eeabf03f61。18替换/1新、联合376源map SHAb32ce8258f75c69515b5dd5afb89c5d2a797021ca68da24b81601f097eda610a，26新CPU仅定义；root只读37来源及19目标无冲突核验exit0，未应用/测试/build/GPU。新supervisor SHA3050d8616dbf9cbf7d6038f6f77dab282fa02cadfe962ce5bc28b5b62d708f89仅接受v2/v3增量，旧21+2不能作为此新脚本通过证据。下步先补既有collection预约内restore-only单次selection接口和最终exe，再构造有限包及完整矩阵；不重跑全calibration或复制2004行record/verify，20/44仍静态建议未登记，见§8.47～8.48。
> 实际会话和冻结锚位于 target/unified-quant-selection-c1-r1；不得重启/重跑已消费尝试。最终结果见规格 §8.19 及各 prepared/result.json。
> 正常CLI CPU预检静态交付SHA6604807b9b508530c635c3fbf5b102c53de12cf0d8ab5a0a4d464ab375af488f，独立窄审39来源稳定、无确定阻断，review SHAb1c656a94d645d6c64e82272134aaed3c67adf639e6b61df616f9252f9fa7093；20新测试仅定义。持久attempt journal静态交付SHA4b369d9a945438eae82d9182053a188d911542b42fe027aa10c68bb2429919db，16测试仅定义，尚无Observer/Completion/record/verify。组合预计363文件，有限五Cargo/90CPU与三模型CPU preflight正在准备，未启动；不重复原106项、不运行GPU。C32选择及Worker单profile池均只有静态交接，无新性能登记或协议部署，见§8.42。
> CLI+journal隔离CPU集成已唯一启动session34943（chunk daba15），redirector78656/实际Python40764；runner SHA8901cc2e4c28ce32b9e5d82c98cd5696f465a3baa48da6b20ac414dcef8e8dd6，plan SHAdec3b920dd68449d34b05c54c55a142fc4cbc93246eaaf88cf61583ca728833d。363源已物化，前两Cargo及45非CUDA测试actual0，CUDA与三模型CPU预检待终态，勿重启入口/改冻结源；没有GPU调用。见§8.42。
> 同一CLI+journal集成session34943已actual exit0（chunk216dbc），90/90 CPU、五Cargo、ONNX三输入的两CPU准备、B11/B15/ONNX三CPU preflight均PASS；result SHAe683b971b4b9bd71bd6270d3547477a1f13ecde1ef888c755292a4ac0440cd7a。363编译源前后SHA2c7cbac9c052a8f93be2d4d5d43533b07966f305644b9e5fb1880261ca50d7d8，4637绑定稳定；新CLI SHA737a13eba4d20ecbc6edf8b6262554f572ebab9a7259b8ecc4fd5852cd7ce009。后续增量基线为target/unified-quant-production-cli-integration-plan-r1/run/snapshot，无须重跑90/原106；0GPU，record/verify尚未实现。新静态设计已识别ModelUpload前设备准入、normal restore启动实际观察点及同campaign双stage单次claim缺口，不能以外层补记或另开consume冒称已接通；见§8.42。
> Durable Observer静态增量已封存于target/unified-quant-execution-observer-static-r1，delivery SHA5df5bb9547f566172108adfd3f81f7c23ebae18622692e029c5f99c54dd12b22，actual363→366，20测试仅定义。包括五头/decoded原bits、finite门、finish返回确认、失败poison及两个日志完整chain终态读回；未应用/构建/测试/GPU。record/verify静态方案SHA5d8c3fbf3341741c9eb8658b316cc078484f93916d62db93dd0918b4246be313；startup hooks和CPU Prepared分别静态在制，StagePermit/实际CLI/新GPU登记尚无，见§8.43。
> Observer独立窄审18来源稳定/无确定阻断，review SHAebfd5a79ea0fb210f441e030800986b3133ec21deb51d79a7b320c80f44c16e4。startup增量delivery SHAb420dbc2b69ff5c5aa335ac20d80281f615de41925bbbf539a97fb1283b3fe2e（2新增/3替换、12测试定义），Prepared抽取delivery SHA9e0702d8b9a09683be440006016619424453bbabb4779f0703fdca4380e03361（2替换、3新测试定义）均source-only封存。预计三增量合成368源，有限五Cargo/84CPU及三模型CPU preflight正在准备，尚未运行；ONNX复用已有prefix3输入。StagePermit/CLI record/verify未实现，不据静态稿新增GPU或重跑90/106。见§8.43。
> 三增量有限CPU集成已唯一启动session29352（chunk656d6e），target/unified-quant-observer-startup-prepared-integration-r1；runner SHA2f86bb6b77a3717ce4238d7c3fa1cb1c0d544cbc793269db347e1d37e3fa0bd8，plan SHA95eda601d759b1a78ed29bf5d0ace6ab7ee10f58e0d4d0367d1961002a5af9e9。368源/五Cargo/84CPU/三真实CPU preflight，首非CUDA库32检查actual0，终态待观察勿重启；ONNX复用prefix3，0GPU。startup独审18来源稳定无确定阻断，review SHAb0ef65f0a6b466e38bff8efd5aa3a934e18a929c7e3fc4f8f45907431a347caf。StagePermit及私有record驱动仅静态在制、CLI仍拒绝record/verify，见§8.43。
> 同一session29352已actual exit0（chunk3eaf3d）：84/84 CPU、五Cargo与B11/B15/ONNX三真实CPU preflight均PASS；result SHA0cecaeb0f74a39a4eb4d32a42946657637aa7f74e06031b556c3c9809d5bb3d8，368编译源前后SHAddb4bf26f7ca7aa31b21cdd20a352fd840aa58ecae2e82037100442b68035770，最终4651绑定稳定。新CLI SHA732e99c12dc94adb2d4fda7d1f1d0ec33955aea0d9368bd14a407c489d477c21；后续基线为该run/snapshot，不重复84/90/106；0GPU、CLI record/verify仍未接通。限定复核确认普通统一后端可用合格FP16 DualFFN，而v1录制禁用导致其包不能部署时临时开启；选择前须补真实一次probe预算，保留最快旧FP16控制。见§8.43。
> StagePermit与私有正常record驱动在新target目录静态在制，尚未封存/测试；root私有stdin gate已source-only交付SHA27206c711e72d20f28b02f309e32c35d55441687e6e0dead76b266eca3e17a8c（2源/8定义，未挂接或执行），claim后仍须核gate与permit Source相等。DualFFN限定7源交接SHAdedc77db8f012a33394524c6fc944c0f7340a77d6dec3b915a38f54c1a20708e：普通统一路径支持合格FP16组，但v1录制禁用；恢复未确认误路由，需在Binding后/首次目录前为真实OnceLock probe做有界记账。0新GPU/性能登记，不改现16类schema或最快旧FP16控制；见§8.44。
> StagePermit record-only静态交付SHA0ca19626a727dcab454301a4a5d305733dcdfe5eb2d71f4ee955e76bced0e695（3替换/2新、18定义）、私有正常record driver交付SHAbf0c922d30e89cca8b5150adae6e448eb3d999a5eba6cb2627c01034d9d0d0ac（1替换/2新、16定义）均已封存，0测试/构建/GPU。绑定actual368，含permit计划372源SHAd0a44889a669357f8ab641a4df222f48852b86be01face05e6d17091a1fa949a；加gate预计374，尚无合成快照/runner。CLI仍硬拒绝record/verify，supervisor/actual exit reader/verify恢复比较未接；不得把静态功能当368已测能力。见§8.44。
> StagePermit独立静态窄审19来源稳定/无确定阻断，review SHA3ebe6ebe3aa765f78714b5052b2ee3180145fa0e8f632ccb60e252a27bfe1cd2，source receipt SHA4592359727c69deb410a0af6060875f626e35c6f0afbc6b1ace1346471ef81e1；结账/Observer既有生命周期主体与actual368逐字节一致。仅source review，18定义未执行；公开CLI/supervisor/verify仍待接线，见§8.44。
> 正常record/verify完整r2接线已封存：permit delivery SHA8c1603dd1b683eca94c8d60e0a21e251b7b1ed858211228be1e3e83d9c8c06bf，CLI delivery SHAf613ea8cd934cba6550710dd98ba0f31a40b0350b633c92d2761792bc4d51e58，supervisor delivery SHA30fb7832b273e2b0ebd33f399764091120435568af3b9daf458928ccadbcfbab。375源有限集成在target/unified-quant-record-verify-integration-r1，唯一session75122（初始chunk3b20fb），runner SHA28523d0d66867dcdc56027092bb010ec6d04316d5444eb0f584333004a099abd，plan SHA21dd3dbf37f089d0688ffec3be0ba63e963891130d49f27eef8ffd2417f18cfa；21mock/2真实CPU子进程已actual exit0，226RustCPU/五Cargo/三模型CPUpreflight终态待观察，勿重启。0GPU，旧原件/正式Worker不改；独立runner窄审无确定阻断但在启动后收口，见§8.45。
> 同一session75122已actual exit1（chunk4d0356），result SHAd42fae8d9ae51d6bd0e0e3673a9959275980de6ed5deaae6b87c4d83033183f2。21mock/2真实CPU child通过，首nonCUDA库编译exit0；permit组29PASS/1FAIL（parent_components_are_rejected_before_creation），余下Rust/CUDA/三模型CPUpreflight未执行。375冻结源稳定，原件不改/入口不重启；Windows verbatim PathBuf.join提前规范化`..`夹具问题待新隔离r3确认，不放宽runtime门，见§8.45。
> 路径夹具r3仅改1测试函数，delivery SHA1da3a0870b3df2826d69d4c0517ae725a52fc13d9f3cddc2eaa8cdefec2525dc；根因为Windows verbatim join规范化，runtime门不变。新target/unified-quant-record-verify-integration-r2唯一启动session98926（chunk811e15），375源/五Cargo/226Rust/三模型CPUpreflight；继承已通过21+2不重跑，受影响permit30重验。runner SHA33b6030829e67bd7ee95f7626ce37b2092abf6b871cbba08101ba43f6591012f，plan SHAa5ed741af13fceca90bc44c34d0b36c6e58c0aec616f3e1bf894b81b7f7402c8，delivery SHAca2b613b9d8c2f63759b2bceb826a788e2c9be516cce4c29ab0b0ff984985356；终态待观察，勿重启/修改冻结源，0GPU，见§8.45。
> 历史方案的完整自动选型、最终模型数值验收仍未完成；NVFP4 为后续可选范围，
> 不能把规划或阶段性检查写成性能收益/认证；正式二进制与原认证计划未替换。

KataGo 围棋引擎的 Rust 移植(基座:KataGo-Lite/katago-rs,约 10 万行,含与 C++ 对拍的 oracle 测试)
+ 针对本机 RTX 5070 Ti(Blackwell SM120)的 CUDA 推理后端。

## 构建与运行

- 构建(默认特性,dummy 后端):`cargo build --workspace`
- TRT 后端:`cargo build -p katago --features trt`(需本机 CUDA + TensorRT 头文件)
- CUDA 后端(手写 kernel,nvcc 编译):`cargo build -p katago --features cuda`;
  冒烟 `cargo test -p kata_nn --test test_cuda --features cuda`
- 可选 INT8 后端（2026-09-22）：同一 `cuda` feature，`nnBackend=cudaint8backend`、
  `cudaInt8Scope=ffn`（默认）或 `transformer`；原生 B11/B15 及模型声明的 FFN
  结构化剪枝。独立有损精度身份，不能复用 FP16 plan；构建、误差和性能边界见
  `docs/RustGoINT8后端.md`。未更换原正式二进制，也未放宽原 FP32 数值门。
  后续剪枝诊断新增 `cudaInt8MinFfnWidth`（默认 0，B15 性能候选 384）、
  大 batch 窄层 warp 量化；结果与 SGF 校准边界见 `docs/RustGoB15剪枝INT8诊断.md`。
  后续 INT8 专项优化：全 FFN 默认融合 RMSNorm/量化，非零宽度阈值保留原路径；
  `KATAGO_CUDA_INT8_RMS_FUSION=0|1` 可诊断覆盖。当前独立构建为
  `target/int8-specialized/build/release/katago-rs.exe`；ABBA、原始输出逐位对拍及
  使用入口见 `docs/RustGoINT8专项优化.md`，不代表精度恢复或棋力认证。
  用户允许本次 INT8 实验改变中间求和精度，仍须独立验证最终输出；
  已采用的 kernel 改写与原量化输出一致，简单输出校准未通过，不得标为精度恢复。
  2026-09-23 新一轮研究与实测：用户允许本轮混合精度采用相对同模型 FP16
  的胜率输出最大偏差 ≤6 个百分点，另报 policy/目差；这是独立实验门，
  不改原 FP32 认证门。q64 注意力与可选逐形状 INT8 GEMM 调优见
  `docs/RustGoINT8注意力与算法调优.md`，网络方案见
  `docs/RustGo低精度优化新方案调研.md`。GEMM 调优默认关闭，
  `KATAGO_CUDA_INT8_GEMM_TUNE=0|1`；启动与性能证据必须分开。
- CUTLASS(B2 dual-FFN 主机侧 CUTLASS DualGemm):build.rs 按
  `KATAGO_CUTLASS_ROOT` → `third_party/cutlass` → `D:/code/cutlass` 查找
  (本机已克隆 v3.9.2 到 D:/code/cutlass);找不到则跳过该 tactic
  (plan 里的 KATAGO_CUDA_DUALFFN=1 静默回退现有路径,不报错)
- 测试:`cargo test --workspace`
- GTP 冒烟:`--model /dev/null`(dummy 后端);真实推理:`--model D:/code/b11fix.onnx
  --override-config nnBackend=cudabackend`(CUDA)或 `nnBackend=trtbackend`(TRT,
  首次构建引擎较慢,plan 缓存在模型旁)
- genconfig(交互式生成 GTP 配置,对齐上游 benchmark.cpp 的 MainCmds::genconfig):
  问答规则/搜索限制/后端(多出 nnBackend 选择写入配置,Rust 端运行时选后端所需)/
  显存缓存/设备,然后 ternary 搜索调优 numSearchThreads 并测半 batch,产物可直接
  供 `gtp --config` 使用;不测"每 GPU 2 个 NN server 线程"(本仓库单消费者凑批设计,
  `NnEvaluator::set_num_threads` 为 no-op)。复用 benchmark.rs 的调优件
  (create_nneval/do_auto_tune_threads 等,pub(crate))
- 数值对拍:`cargo test -p kata_nn --test dump_nn_io_cuda --features cuda --release`
  生成转储 → `.venv/Scripts/python.exe scripts/compare_nn_output.py
  crates/kata_nn/target/nn_io_dump_cuda`(ORT FP32 黄金参考;gates:policy 5e-2、
  value 2.5e-2、misc 1e-2、ownership 5e-3、policy top-1 100%)
- 性能基准(release):`./target/release/katago-rs.exe benchmark --config
  configs/gtp_smoke.cfg --model D:/code/b11fix.onnx --override-config
  nnBackend=cudabackend --override-config numSearchThreads=1 -v 20 -n 1 -t 1,4`
  (不带 -t 时走 auto-tune;模型只加载一次但逐档搜索仍慢,
  务必用 -t 指定 1-2 个配置)
- 日常自动配置（2026-09-20）：`./scripts/tune_rustgo.ps1 -Mode local` 或
  `-Mode worker -Capacity 32`；匹配已有完整认证 profile，ABBA + 1% 收益/
  5% 波动门，产出独立 cfg、计划副本、启动脚本和报告，不改认证身份或 kernel。
  Worker 用临时本机真实 gRPC，无需生产 Server；无匹配 plan 仅基础配置。
  入口说明见 `docs/RustGo自动配置优化.md`、`docs/RustGo运行模式说明.md`。
- 跨机 FULL AUTOTUNE（原生 TF3 v17）：`./scripts/full_autotune.ps1 -Model <模型.bin.gz>`；
  新建本机 plan，先可携带 C++ FP32 金标/实际路径门再 ABBA，最终复验 + READY 发布。
  `--groups` 缩小范围会标 `CUSTOM_GROUPS`；未通过金标不得测速，不能伪造/改写已有认证身份。
  使用、参考包导出及打包见 `docs/RustGo-FULL-AUTOTUNE.md`；ONNX FULL 不支持。
- 历史内核实验 autotune:`python scripts/autotune.py --threads both`(8 决策组 ABBA,
  产出 plans/best-tactic-plan.json + 可直接 `gtp --config` 使用的
  configs/gtp_autotuned.cfg——按 nnEvals/s 线程扫描选 numSearchThreads 并做
  GTP genmove 冒烟验证;`--model/--out-plan/--out-cfg/--threads-sweep` 可改目标,
  `--out-cfg ""` 回到只产 plan 的旧行为);接入选认证 plan:`--override-config
  nnBackend=cudabackend,cudaTacticPlan=D:/code/Rust_KataGo/plans/best-tactic-plan.json`
  (fail-closed:指纹/模型不匹配即报错);设备指纹:`katago-rs cuda-fingerprint --model <f>`
- per-layer 剖析:`KATAGO_CUDA_PROFILE=1` + GTP `kata-raw-nn all`(逐层/attention
  子段耗时,graph 模式下被跳过需配 `KATAGO_CUDA_NOGRAPH=1`);输入 dump:
  `KATAGO_CUDA_DUMP_INPUT=<dir>`;逐层 dump:`KATAGO_CUDA_DEBUG_LAYER=<i>`
- nnbench(固定物理 batch 吞吐,对齐 C++ benchmarknn):`./target/release/katago-rs.exe
  nnbench --model D:/code/b11fix.onnx --override-config nnBackend=cudabackend
  --mode eval --batch 1,2,4,8,12,16,24,32 --iterations 400`(`--mode direct`=
  CudaModel::apply 直连含拷贝,`--mode kernel`=纯前向;eval 的 --workers 默认
  2×当前 batch——serve target 是发射阈值非上限,worker 过多会把 avgBatch 抬高);
  cuBLASLt 候选池探针:`cargo test -p kata_nn --test probe_cublaslt_algos
  --features cuda --release -- --nocapture`
- tactic 组合验证(仿官方 runcudaopttests.sh,对齐 v1.18.1 审计结论):
  `scripts/validate_cuda_tactics.py`(24 case:路径标记断言 + plan
  fail-closed 负例;`--with-numeric` 加数值门——每组合 dump+ORT 对拍)。
  所有 tactic 路径决策点会打一次性 `[cuda-tactic]` 标记(dual_ffn/
  attention/gemm kind×engine/splitk/fusion/rms/padbatch/cublaslt_rank/
  graph),供脚本与人工确认开关真实生效(M2 DUALFFN 断裂先例的制度化)
- graph 回归测试:`cargo test -p kata_nn --test graph_launch_repro
  --features cuda --release`(2026-08-24 修复:无 pre-capture warm 时
  首次捕获的 graph exec 会被后续同流捕获作废,首批发射恒
  CUDA_ERROR_INVALID_VALUE——warmup/首批 eval 静默丢批;
  `REPRO_NO_WARM=1` 可复现)

## 后端选择

- Go Server Worker：`katago-rs nnworker --server 127.0.0.1:50051
  --worker-id rustgo-5070ti
  --model D:/Go/Server/models/kata1-tf3-b11c768-s11001M-d5973M.bin.gz
  --config configs/worker_cuda.cfg --capacity 32`；真实后端当前仅 CUDA。
  `crates/kata_worker` 实现纯 NN gRPC Worker，Server 持有搜索图；协议与
  `D:/Go/Server/proto/worker.proto` 同步。接入/模型身份边界与验证见
  `docs/Go-Server-Worker.md`。合成测试必须显式 `--allow-dummy` + `/dev/null`。
  原生 TF3 v17 直接解析并 lower 为 CUDA 层图，SHA 绑定原始压缩文件；
  `worker_cuda.cfg` 不引用旧 b11fix ONNX 的认证 plan。

- 配置键 `nnBackend`(每模型可用 `nnBackend{i}` 覆盖):`dummybackend`(默认)/
  `trtbackend` / `cudabackend` / `eigenbackend`(未实现,选择即报错)
- `cudabackend` 需要 katago 的 `cuda` feature(透传 `kata_nn/cuda` +
  `kata_program/cuda`);CUDA 后端仅支持 19 路、强制 NCHW 输入
- 命令行示例:`--override-config nnBackend=cudabackend`

## 目录

- `crates/kata_*`:引擎各层(core/game/data/search/nn/program/book/distributed);
  `crates/katago` 为 CLI(bin `katago-rs`)
- `crates/xuanping`:玄枰 GUI(egui/eframe 0.36,冷感东方风;视图:
  Play(分析覆层 `a`)/ Review `r` / Settings「器」`e` / 帮助「键」`?`,
  棋谱条 `k`、玄墨/雪宣双主题 `t`、←/→ 步进;
  运行 `cargo run -p xuanping`(无模型=演示模式);引擎模式
  `cargo run -p xuanping --features cuda -- --model D:/code/b11fix.onnx
  --backend cuda --threads 8`(同 workspace kata_* 直连,AsyncBot 分析回调
  0.25s 喂候选/领地/根胜率,用户执黑引擎执白,中国规则 komi 7.5);
  帧截图自验 `XUANPING_SHOT=<ppm>`(配 `XUANPING_VIEW/THEME/HELP/MODEL/
  BACKEND/THREADS` 截任意状态)、`XUANPING_DEBUG=1` 打布局指标;
  设计稿与生成脚本在 `docs/design/`、`scripts/design/xuanping_mockup.py`)
- `crates/kata_nn`:NN 抽象层与后端(`backends/`:dummy、trt、cuda/cuda_exec);
  `cuda-kernels/*.cu` 为手写 kernel(basic/gemm/gemm_v2/elementwise/attention/
  executor),由 build.rs 按 `configs/sm-targets.json`(sm_120 优先)用 nvcc
  编译成 PTX 嵌入
- `cpp-shim/`:TensorRT C++ FFI shim
- `configs/`:sm-targets.json、gtp_smoke.cfg、gtp_cuda.cfg、gtp_trt.cfg、
  gtp_benchmark.cfg
- `scripts/`:冒烟/对拍/对局脚本(compare_nn_output.py、play_match.py、autotune.py)
- `docs/`:需求汇总.md(含进度 §9)、cuda-fork-parity-plan.md(**对齐/超越
  fork 主规划,跨对话对齐入口,CUDA 性能动工前必读其 §1/§2/§8**)、
  cuda-optimization-plan.md(M4 路线图与 SM120 时效资料)、
  fork-sm120-kernel-notes.md(KataGomo_fork 认证 plan 提炼)、
  搜索算法调研与改进计划.md(kata_search 现状审计、官方 KataGo 演进与学术前沿、
  搜索树改进 P0-P2 路线;含 wideRootNoise 随机项/pondering/subtreeValueBias 写侧/
  evalCache 四处移植断裂清单)、使用与GUI接入.md(GTP GUI 接入与配置用法)
- `plans/`:autotune 产物(best-tactic-plan.json、autotune-history.json)

## 约定

- 仅支持 19 路标准棋盘
- 数值纪律:激活/归约一律 FP32 计算、half 存储边界与官方 KataGo 逐位对齐
  (正确性门前提,见 `docs/需求汇总.md` §7);SiLU=x/(1+exp(-x)) 精确形式;
  禁 fast_math
- 后端路线:手写 CUDA C++/PTX kernel 为主线(nvcc 编译 + cudarc 加载),
  TensorRT 兜底,plan JSON fail-closed
- tactic 开关(KATAGO_CUDA_* 环境变量)一律经 `tactic_plan::tactic_var()`
  读取(优先级 plan > env > 默认;直接 env::var 会绕过认证 plan);
  默认值变更必须 autotune ABBA + 对拍双证据。当前 plan(r1)仅启用
  `KATAGO_CUDA_DUALFFN=1`(CUTLASS DualGemm + SwiGLU epilogue;M2 落地,
  2026-08-17 修复提交断裂后实测 +26.5% eval B16 / +28.0% 搜索 t=32);
  `KATAGO_CUDA_CUBLASLT_RANK=time` 曾于 M1 采纳(+2.55%),2026-08-17
  复审下架(B2 接管 ffn_up 后边际归零,搜索口径净负 -4%)
- cuda-host 源码编译失败 = 构建失败(fail-loud,CUTLASS 缺失仍静默跳过;
  逃生门 KATAGO_ALLOW_BROKEN_CUDA_HOST=1)——M2 曾提交编译不过的
  dual_ffn_cutlass.cu 被 build.rs 静默跳过,DUALFFN 空转数小时无人察觉
- 依赖真实模型的测试用 `KATAGO_TEST_MODEL_DIR` 环境变量定位模型,缺失时自动跳过
- 新后端实现需实现 `kata_nn::backend::Backend` trait 并接入 `kata_program::setup`
  的 backend 选择
- 性能变更必须 ABBA 实测(基准命令见上),慢于基线即回退;数值变更必须过
  整图对拍(compare_nn_output.py RESULT: PASS);启用休眠代码路径(如
  长期未跑的 tactic 组合)前同样先过对拍(FUSION=none 的 act384f16
  断裂 bug 即先例)
