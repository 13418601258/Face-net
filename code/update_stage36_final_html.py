"""Update only S3.6, preserving UTF-8 and historical experiment records."""
from pathlib import Path
import argparse
import json
import re
import hashlib

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT/'log/方向/实验计划-阶段0至3.html'
OUT = ROOT/'code/reports/stage3_s36_v5_final'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',action='store_true');args=parser.parse_args()
    raw=HTML.read_bytes();text=raw.decode('utf-8-sig')
    if '\ufffd' in text: raise ValueError('Existing replacement character; refuse update')
    OUT.mkdir(parents=True,exist_ok=True)
    backup=OUT/'s36_before_final.html'
    if not backup.exists(): backup.write_bytes(raw)
    start=text.index('          <div class="step" data-id="s36">')
    end=text.index('        </section>',start)
    old=text[start:end]
    archive=OUT/'s36_v1_v4_history.html'
    if not archive.exists():
        archive.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><title>S3.6 v1-v4 历史记录</title><base href="../../../log/方向/"></head><body><h1>S3.6 历史计划与结果：已归档，不再作为活动计划</h1>'+old+'</body></html>',encoding='utf-8')
    status='最后一次验算 · 结果待计算'
    results='<p><strong>执行状态：</strong>协议已冻结，尚未计算，不提前宣称通过或失败。</p>'
    if args.results:
        s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
        status='第一关通过 · 可进入第二关' if s['network_eligible'] else '第一关结束 · 不进入第二关 · MDL 对应路线停止投入'
        rows=''.join('<tr><td>'+r['layout']+'</td><td>'+str(r['feasible'])+'/'+str(r['expected'])+'</td><td>'+str(r['joint_selected'])+'</td><td>'+format(100*r['median_relative_gain'],'.5f')+'%</td><td>'+('通过' if r['passed'] else '未通过')+'</td></tr>' for r in s['decisions'])
        results='<p><strong>最终决定：</strong>'+s['decision_zh']+'</p><p>'+s['result_zh']+'</p><table class="tbl"><thead><tr><th>布局</th><th>可行扰动条件</th><th>选择联合项</th><th>相对普通正则收益中位数</th><th>冻结条件</th></tr></thead><tbody>'+rows+'</tbody></table><p><strong>解释边界：</strong>这是当前数据、局部参数化和预算下的最终立项决定，不是所有 MDL 方法不可能有效的证明；不把数值验收通过当作研究通过。</p>'
    block='''          <div class="step" data-id="s36">
            <input type="checkbox" class="checkbox" aria-label="标记 S3.6 完成" />
            <div class="step-top"><span class="step-id">S3.6</span><span class="step-title">第一关最后一次验算：身份—表情联合紧凑项</span><span class="tag key">STATUS</span></div>
            <div class="step-body">
              <p><strong>范围收缩（2026-10-19）：</strong>v1–v4 已完成当前单谱路线的实现、几何、稀疏可辨识性、权重和额外信息检查；v4 三种布局的选择后额外收益中位数均为零。停止重复扩展原门控、权重和步长网格。本轮只回答：联合身份与表情结构能否提供普通正则没有的可选择纠错收益。</p>
              <p><strong>最终停止约定：</strong>这是第一关最后一次数据验算。新机制未达到冻结条件，就结束 MDL 驱动对应这条支线，不进入第二关 MLP，不再增加 v6、扩数据或改阈值；不会自动改成压缩论文来维持投入。通过也只允许小型网络验证，不等于已取得网络收益。</p>
              <h4>唯一活动计划</h4>
              <ol>
                <li><strong>先核验定义：</strong>20 身份 × 7 状态共同刚性 GPA，只去平移旋转、保留尺度；联合减去一个总均值。身份/表情协方差分别为 20×20、7×7，全谱计入零值并报告实际秩，不套用单状态 K=19。检查共同身份置换不变性、独立打乱身份配对的响应、有限差分和直接/局部 Gram 一致性。</li>
                <li><strong>公平恢复对照：</strong>复用三处局部区域，共同调和图域和六参数场。原始曲面采样、固定边界、双拉普拉斯网格正则和观测约束一致；比较普通正则、普通正则+逐状态谱、普通正则+联合双谱。能量梯度比仅 0.1/1，步长上限仅 0.0025/0.01；不看真值选步长或候选。</li>
                <li><strong>仅保留关键条件：</strong>独立表情错配、同身份跨表情一致错配，两锚点/六分散锚点，无噪/σ=10⁻⁵；三种子共24个扰动队列，每队列140张人脸，另6个不重复的正确对照。每张脸12个选择、12个独立审计锚点；稠密评价排除所有锚点。无效初值全部记录。</li>
                <li><strong>预先判定：</strong>局部有效率≥80%；联合项实际被选比例≥20%；相对普通正则收益中位数&gt;1%且绝对 NRMSE 改善&gt;10⁻⁹；两类错配分别改善，含噪条件改善，至少5/7状态、2/3种子方向一致，独立审计改善，另一档步长仍有收益；逐身份实质损害≤20%，方差不低于原基线99%，正确对照选择后漂移≤10⁻⁹，几何约束通过。以上为项目投入判据，不是统计显著性声明。</li>
                <li><strong>仅通过后做同轮闭环：</strong>最多80步，使用冻结目标和可观察的停止条件，达到上限不能写成收敛。还须≥80%条件收敛、最终相对普通正则收益&gt;1%并通过几何/漂移保护，才进入第二关。局部门禁失败则本轮立即结束该支线。</li>
              </ol>
              <p><strong>与原论文的关系：</strong>核验 §4 的双谱与双拉普拉斯机制，仍采用局部六参数场和原三角片采样，而非全脸 TPS/L-BFGS，因此是核心机制对照，不称为完整复现。共同图域来自已对齐代理基线，固定 GPA 未包含迭代重对齐，所有身份已观察，锚点为合成观测，均须保留在结论边界。</p>
              <h4>最终计算与决定</h4>
              RESULTS
              <p><strong>活动产物：</strong><a href="../../code/stage36_final_protocol.json">冻结前计划</a> · <a href="../../code/reports/stage3_s36_v5_final/">最终验算目录</a>。完成后保存协议与源文件哈希、逐条件/逐身份结果、参数、结构检查、数值检查和图表。</p>
              <details class="fold"><summary>历史证据（已完成，不再追加实验）</summary><div class="fold-body">
                <p>v1：正确基线被推离且局部折叠；v2：平滑场消除部分几何问题，但稀疏零空间保留错配；v3：排除单纯迭代不足与小权重解释，暴露含噪残差门控问题；v4：相对普通修正平滑无稳定、可选择附加收益。</p>
                <p><a href="../../code/reports/stage3_s36_v5_final/s36_v1_v4_history.html">完整历史卡片</a> · <a href="../../code/reports/stage3_s36_v4_information/summary.json">v4 主结果</a> · <a href="../../code/reports/stage3_s36_v4_information/review.json">v4 独立复核</a>。原代码、结果、冻结协议不覆盖。</p>
              </div></details>
            </div>
          </div>
'''
    block=block.replace('STATUS',status).replace('RESULTS',results)
    if args.results:
        review=json.loads((OUT/'review.json').read_text(encoding='utf-8'))
        verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
        if not verification['passed']:raise ValueError('Independent verification has not passed')
        extra='<p><strong>关键读数：</strong>联合谱能够检测各表情内独立打乱身份配对，分数上升约1.070–1.097，但两档步长的72条接受联合更新没有一次降低稠密误差（含重复条件，不作72份独立证据）。选择器在全部24条扰动队列中均未选择联合项；主步长固定候选池的事后真值最优也无实质收益。普通正则策略本轮亦选择保持。</p><p><strong>去重与空跑说明：</strong>六锚点满秩解析解消除了初始扰动差别；30条记录对应21个不同起点，24条扰动记录只有18个不同起点，其中15个非数值零误差起点。六锚点无噪声条件是无漂移对照，不作为纠错失败的主要证据。主步长去重后，两锚点24个联合候选全部变差，误差增幅中位数0.437%；六锚点6个含噪联合候选全部变差，中位数10.653%。百分比须结合绝对误差，不能跨量级夸大。</p><p><strong>验收：</strong>9项数学测试、18项实数据导数检查通过；独立'+str(verification['check_count'])+'项检查通过，180次选择已重放、90个主步长端点已重算。按冻结协议不启动含噪闭环和第二关 MLP，第一关验算已结束。</p><p><a href="../../code/reports/stage3_s36_v5_final/final_report.md">最终简报</a> · <a href="../../code/reports/stage3_s36_v5_final/summary.json">最终判定</a> · <a href="../../code/reports/stage3_s36_v5_final/review.json">去重与候选复核</a> · <a href="../../code/reports/stage3_s36_v5_final/verification.json">独立验收</a> · <a href="../../code/reports/stage3_s36_v5_final/final_decision.png">结果图</a> · <a href="../../code/reports/stage3_s36_v5_final/final_decision.pdf">PDF图</a></p>'
        block=block.replace('<p><strong>活动产物：</strong>',extra+'<p><strong>活动产物：</strong>')
        block=block.replace('aria-label="标记 S3.6 完成"','aria-label="标记 S3.6 完成" checked')
    updated=text[:start]+block+text[end:]
    if '\ufffd' in updated: raise ValueError('Replacement character generated')
    if updated.count('<section')!=updated.count('</section>'): raise ValueError('Section mismatch')
    if updated.count('<details')!=updated.count('</details>'): raise ValueError('Details mismatch')
    HTML.write_bytes(updated.encode('utf-8'))
    check=HTML.read_bytes();assert not check.startswith(b'\xef\xbb\xbf');assert check.decode('utf-8')==updated
    print(json.dumps(dict(encoding='UTF-8 without BOM',before_bytes=len(raw),after_bytes=len(check),s36_before_lines=len(old.splitlines()),s36_after_lines=len(block.splitlines()),sha256=hashlib.sha256(check).hexdigest()),ensure_ascii=False))

if __name__=='__main__':main()
