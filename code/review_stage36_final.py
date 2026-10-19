"""Review saved final evidence, deduplicate full-rank starts, no new experiment."""
import json
from pathlib import Path
import numpy as np
from stage36_final import OUT,ROOT,STATES
from stage36_smooth import sha,write_json

def main():
    jobs=[json.loads((OUT/'checkpoints'/f'seed_{s}.json').read_text(encoding='utf-8')) for s in range(3)]
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    verification=json.loads((OUT/'verification.json').read_text(encoding='utf-8'))
    groups=[];comparisons=[];maximum_equal_difference=0.
    for job in jobs:
        seed=job['seed']
        with np.load(OUT/'parameters'/f'seed_{seed}.npz') as z:
            representatives=[]
            for start in job['starts']:
                case=start['case'];match=None
                for g in representatives:
                    rep=g['representative']
                    if start['layout']!=g['layout'] or start['sigma']!=g['sigma']:continue
                    distance=float(np.max(np.abs(z[case+'_start']-z[rep+'_start'])))
                    if distance<=1e-12 and np.array_equal(z[case+'_fit'],z[rep+'_fit']) and np.array_equal(z[case+'_selection'],z[rep+'_selection']) and np.array_equal(z[case+'_audit'],z[rep+'_audit']):
                        match=g;maximum_equal_difference=max(maximum_equal_difference,distance);break
                if match is None:
                    match=dict(seed=seed,layout=start['layout'],sigma=start['sigma'],representative=case,members=[]);representatives.append(match)
                match['members'].append(case)
            groups+=representatives
    rows=sum([j['candidates'] for j in jobs],[]);policies=sum([j['policies'] for j in jobs],[])
    # All two-anchor conditions remain distinct; six-anchor patterns share the exact optimum.
    unique_corrupted=[r for r in policies if r['pattern']!='control' and r['policy']=='joint' and r['cap']==.01 and (r['layout']=='two_v2' or r['pattern']=='independent')]
    informative=[r for r in unique_corrupted if r['initial_nrmse']>1e-9]
    for layout in ['two_v2','six_spread']:
        for family in ['regularizer','single','joint']:
            rr=[r for r in rows if r['layout']==layout and r['family']==family and r['pattern']!='control' and r['cap']==.01 and r['accepted'] and (layout=='two_v2' or r['pattern']=='independent')]
            gains=[r['initial_nrmse']-r['nrmse'] for r in rr]
            comparisons.append(dict(layout=layout,family=family,unique_primary_accepted=len(rr),improved=sum(g>1e-12 for g in gains),worsened=sum(g< -1e-12 for g in gains),median_relative_recovery_gain=float(np.median([g/max(r['initial_nrmse'],1e-12) for r,g in zip(rr,gains)])) if rr else None))
    strict=[]
    for job in jobs:
        for r in job['numeric']:
            rel=min(r['errors'])/max(abs(r['analytic']),1e-25)
            strict.append(dict(seed=job['seed'],term=r['term'],direction=r['direction'],relative_error=rel,passed=rel<1e-3))
    # This tighter derivative audit is diagnostic only; it cannot relax the frozen decision.
    candidate_joint=[r for r in rows if r['family']=='joint' and r['pattern']!='control' and r['accepted']]
    result=dict(counts=dict(recorded_starts=30,distinct_starts_including_controls=len(groups),distinct_corrupted_starts=len(unique_corrupted),distinct_nonfloor_corrupted_starts=len(informative)),
        deduplicated_groups=groups,maximum_equivalent_start_difference=maximum_equal_difference,
        strict_gradient_checks_passed=all(r['passed'] for r in strict),strict_gradient_checks=strict,
        joint_accepted_raw_count=len(candidate_joint),joint_accepted_raw_improves=sum(r['nrmse']<r['initial_nrmse']-1e-12 for r in candidate_joint),
        joint_accepted_raw_worsens=sum(r['nrmse']>r['initial_nrmse']+1e-12 for r in candidate_joint),
        unique_primary_comparisons=comparisons,distinct_informative_selected_joint=sum(r['chosen_family']=='joint' for r in informative),
        joint_pairing_deltas=[j['structure']['joint_pairing_delta'] for j in jobs],
        independent_verification_passed=verification['passed'],deduplicated_decision_unchanged=all(r['absolute_gain']==0 for r in unique_corrupted),
        interpretation='The joint statistic detects identity-expression pairing, but no tested accepted joint candidate reduces dense correspondence error from the exact sparse fit. Both prior and compactness can bias an already good fit. Full-rank starts repeat across corruption patterns; repetitions are not independent evidence. This local test does not prove distant-start/global impossibility.')
    write_json(OUT/'review.json',result)
    lines=[
        '# S3.6 第一关最后一次验算',
        '',
        '最终决定：结束 MDL 驱动稠密对应这条研究支线，不进入第二关 MLP，不追加权重搜索、数据规模或新的验算版本。',
        '',
        '## 已完成',
        '',
        '- 仅使用已解压数据的缓存：ast001–020、七个状态，每张 33390 顶点；140 张共同刚性 GPA，四次迭代收敛，不归一化个体尺度。',
        '- 核验身份—表情联合 log 谱：身份 20 项、表情 7 项，实际秩均满秩；当前单状态对照仍用 K=19。',
        '- 统一局部调和图域、六参数场、原三角片采样、边界固定和双拉普拉斯正则；两类扰动、两种锚点布局、两档噪声、三处区域。',
        '- 30 条初值记录全部几何有效；300 个候选、180 个策略结果、25200 条逐脸评价。210 个候选被接受，90 个因精确零空间无方向而保持，不能把保持算作纠错成功。',
        '',
        '## 核心结果',
        '',
        '- 联合目标对独立打乱各表情身份配对有明确响应，分数上升约 1.070–1.097；共同置换不改变分数，逐状态谱对独立身份重排也不变。说明新目标确实利用了跨表情配对结构。',
        '- 两锚点和六分散锚点的含联合项候选均未被观测选择器选中，相对可执行普通正则策略的收益中位数都是零；两档步长一致。普通正则策略本轮也均选择保持原状。',
        '- 在主步长固定候选池中，即使用稠密代理真值事后挑选联合候选，也没有达到“绝对 NRMSE 改善 >1e-9 且相对改善 >1%”的机会；真值挑选仅作诊断，不作为方法。',
        '- 两档步长共72条接受的含联合项扰动更新全部增加稠密对应误差；这些包含六锚点重复条件，不能解释为72份独立证据。',
        '- 去重后主步长：两锚点24个联合候选均变差，误差相对初始值增加中位数约0.437%；六锚点6个非重复含噪联合候选均变差，增加中位数约10.653%。近零误差条件的百分比敏感，应结合 CSV 中绝对误差。',
        '',
        '## 重复、空方向与数值核验',
        '',
        '- 六锚点设计满秩，精确最小二乘会消除两类初始扰动的差别：同种子/噪声下起点相同至约1.74e-15。30条记录合并后为21个不同起点；24条扰动记录中有18个不同起点，其中15个非数值零误差起点。结果没有因此改变。',
        '- 六锚点无噪声条件已精确恢复且无可行零空间，是无漂移对照，不是支持“联合项纠错失败”的主要证据。真正的失败证据来自欠定与含噪起点。',
        '- 9项数学测试、18项实数据方向导数检查通过；补充严格相对误差检查也全部小于1e-3。独立853项检查通过，重放全部180次观测选择，重新计算90个主步长选择端点。',
        '',
        '## 结论适用范围',
        '',
        '该路线能检测部分统计结构，但在最后一次预定局部恢复测试中，没有转化成额外、可选择的对应改善，因此不值得继续为“驱动网络获得更好对应”投入。第一关验算完成，研究关卡未通过；按冻结协议跳过多步闭环和第二关。',
        '',
        '这不是全部 MDL 的不可行性证明，也不是原论文完整复现：当前是局部六参数场、原三角片表面、固定共同 GPA、已观察身份及合成代理锚点，未实现全脸 TPS/L-BFGS 和独立解剖真值。按本项目最终投入约定停止，不以这些边界继续追加实验。',
        '',
        '冻结协议、summary.json、review.json、verification.json、CSV、参数和 final_decision.png/pdf 位于本目录。'
    ]
    (OUT/'final_report.md').write_text(chr(10).join(lines)+chr(10),encoding='utf-8')
    write_json(OUT/'review_manifest.json',dict(inputs={p.name:sha(p) for p in [OUT/'summary.json',OUT/'verification.json',OUT/'structure_checks.json',OUT/'candidates.csv',OUT/'policies.csv']},outputs={p.name:sha(p) for p in [OUT/'review.json',OUT/'final_report.md']},source_sha256=sha(Path(__file__))))
    print(json.dumps({k:v for k,v in result.items() if k not in ['deduplicated_groups','strict_gradient_checks']},ensure_ascii=False))

if __name__=='__main__':main()
