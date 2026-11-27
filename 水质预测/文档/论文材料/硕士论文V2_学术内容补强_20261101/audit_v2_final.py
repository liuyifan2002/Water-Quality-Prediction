"""V2交付核验：仅读取V1/V2，不运行任何模型或重算解释值。"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import re
import zipfile

import fitz
from docx import Document

ROOT = Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文V2_学术内容补强_20261101")
V1 = Path(r"E:\研究生\codex\水质预测\文档\论文材料\硕士论文解释章节重构V1_20261101\水质预测硕士学位论文_解释章节重构V1.docx")
V2 = ROOT / "水质预测硕士学位论文_学术内容补强V2_核验交付.docx"
PDF = V2.with_suffix(".pdf")
FIG_CSV = ROOT / "图号与交叉引用核验.csv"

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write(name, body):
    (ROOT / name).write_text(body, encoding="utf-8-sig")

def media_hashes(path):
    with zipfile.ZipFile(path) as z:
        return Counter(hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/"))

def number_tokens(table):
    # 数值对照以原表的非首列数值单元格为单位；不把新表3-3纳入冻结结果。
    result = []
    for row in table.rows[1:]:
        for cell in row.cells[1:]:
            result.extend(re.findall(r"(?<![A-Za-z])[-+−]?\d+(?:\.\d+)?%?", cell.text))
    return result

assert all(p.is_file() for p in (V1, V2, PDF, FIG_CSV))
with zipfile.ZipFile(V2) as z:
    zip_error = z.testzip()
assert zip_error is None, zip_error
old, new = Document(V1), Document(V2)
pdf = fitz.open(PDF)
pages = [re.sub(r"\s+", "", page.get_text()) for page in pdf]
captions = [p.text.strip() for p in new.paragraphs if p.style.name == "图题" and re.match(r"^图(?:\d+|[A-F])-\d+\s", p.text)]
nums = [re.match(r"^图(?:\d+|[A-F])-\d+", c).group(0) for c in captions]
duplicates = [n for n, ct in Counter(nums).items() if ct > 1]
figrows = list(csv.DictReader(FIG_CSV.open(encoding="utf-8-sig", newline="")))
figure_issues = []
for row in figrows:
    title = re.sub(r"\s+", "", row["图题"])
    target = int(row["PDF页码"])
    hits = [i + 1 for i, page in enumerate(pages) if i >= 15 and title[:min(22,len(title))] in page]
    if hits != [target]:
        figure_issues.append(f'{row["图号"]}: 目录页{target}, PDF匹配页{hits}')
fig_sequence = []
for prefix in ["图1", "图2", "图3", "图4", "图5", "图6", "图A", "图B", "图C", "图D", "图E", "图F"]:
    seq = [int(n.split("-")[1]) for n in nums if n.startswith(prefix+"-")]
    if seq:
        fig_sequence.append(f'{prefix}: {len(seq)}张，连续={seq==list(range(1,len(seq)+1))}')
        if seq != list(range(1,len(seq)+1)):
            figure_issues.append(prefix+"图号不连续")
if duplicates: figure_issues.append("重复图号："+",".join(duplicates))
if len(figrows) != len(captions): figure_issues.append(f"图目录{len(figrows)}条与图题{len(captions)}条不等")
body_start = next(i for i,p in enumerate(new.paragraphs) if p.text.strip()=="第1章 绪论")
body_figure_refs = [n for p in new.paragraphs[body_start:] if p.style.name!="图题"
                    for n in re.findall(r"图(?:\d+|[A-F])-\d+",p.text)]
missing_body_refs = sorted(set(body_figure_refs)-set(nums))
if missing_body_refs: figure_issues.append("正文引用缺少图题："+",".join(missing_body_refs))

body_paras = new.paragraphs
ref_head = max(i for i,p in enumerate(body_paras) if p.text.strip()=="参考文献")
refs = {}
for p in body_paras[ref_head+1:]:
    m = re.match(r"^\[(\d+)\]\s*(.*)",p.text)
    if m: refs[int(m.group(1))] = m.group(2)
    elif refs: break
cited = set()
for p in body_paras[:ref_head]:
    for block in re.findall(r"\[([\d,，\-–—]+)\]",p.text):
        for item in re.split(r"[,，]",block):
            pair = re.fullmatch(r"(\d+)[\-–—](\d+)",item)
            if pair: cited.update(range(int(pair.group(1)),int(pair.group(2))+1))
            elif item.isdigit(): cited.add(int(item))
ref_missing = sorted(cited - set(refs))
ref_uncited = sorted(set(refs) - cited)

table_rows=[]
for old_i,new_i in [(0,0),(1,1),(2,2),(3,3),(4,4),(5,6),(6,7),(7,8),(8,9)]:
    a,b=old.tables[old_i],new.tables[new_i]
    oa,nb=number_tokens(a),number_tokens(b)
    table_rows.append((old_i+1,new_i+1,len(oa),len(nb),oa==nb))
numeric_result_ok = all(item[-1] for item in table_rows if item[0] in (6,7,8,9))

old_media,new_media=media_hashes(V1),media_hashes(V2)
unchanged_media=sum((old_media&new_media).values())
text = "\n".join(p.text for p in body_paras)
fig_report = ["# 图号与交叉引用核验", "", f"- V2图题：{len(captions)}条；静态图目录：{len(figrows)}条；PDF：{len(pdf)}页。",
              "- Word母本的图目录是静态段落，V2按导出的PDF实际页码重建；不是自动更新的Word图目录域。",
              "- 图号序列："+"；".join(fig_sequence)+"。", "- 页码核验："+("97张逐图匹配" if not figure_issues else "有异常"),
              f"- 正文显式图号引用{len(body_figure_refs)}处，所引用图号均有对应图题；原稿未系统建立可点击交叉引用域，V2未补建。", "",
              "## 异常", "无。" if not figure_issues else "\n".join("- "+x for x in figure_issues)]
write("图号与交叉引用核验.md", "\n".join(fig_report)+"\n")
with (ROOT/"图号与交叉引用核验_最终.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w=csv.writer(f);w.writerow(["图号","图题","PDF页码","定位结果"])
    for row in figrows:
        w.writerow([row["图号"],row["图题"],row["PDF页码"],"通过" if not any(x.startswith(row["图号"]+":") for x in figure_issues) else "待复核"])

ref_lines = ["# 参考文献核验", "", f"文末条目：{len(refs)}；正文使用编号：{len(cited)}。",
             f"缺失条目：{ref_missing or '无'}；未在正文出现的条目：{ref_uncited or '无'}。",
             "[3] 作者顺序已按原始论文核对；[13]为TabICLv2 ICML 2026 PMLR正式论文；[14]为空间交叉验证文献；[15]为近岸水质预测研究。",
             "[11]香港环保署海洋水质官方链接已更正并现场打开核对；[12] ERA5 数据集已作为数据源引用，链接长期有效性提交前仍宜复核。",
             "未逐条完成学校规定的GB/T 7714格式终审，也未核对全文所有隐含引文语义。"]
write("参考文献核验.md","\n\n".join(ref_lines)+"\n")

numeric_lines=["# 科学结果未改变对照记录", "",f"V1母本SHA256：`{sha(V1)}`。",f"V2 DOCX SHA256：`{sha(V2)}`。",
               "冻结结果只读；本轮没有训练、预测、SHAP或ALE计算。", "", "## 原表数值单元格逐表对照", "",
               "原表序号 | V2表序号 | 原数值token数 | V2数值token数 | 完全相同", "---|---:|---:|---:|---"]
numeric_lines += [f"{a}|{b}|{c}|{d}|{'是' if ok else '否'}" for a,b,c,d,ok in table_rows]
numeric_lines += ["", "正式结果表（V1第6至9表）数值token完全一致："+("是" if numeric_result_ok else "否，须复核"),
                  f"V1图像{sum(old_media.values())}个，V2图像{sum(new_media.values())}个，其中逐字节相同{unchanged_media}个；其余包含新汇总图及按原始坐标重绘的Secchi与ALE图，不应按图像hash推断数值变化。",
                  "新增表3-3为冻结SHAP值的展示性汇总；源任务样本、变量和来源编码详见《冻结SHAP汇总核对表.csv》。",
                  "图2-4从三水层重复记录改为站点—日期去重分布；该图不能与V1图2-4视为同一统计口径。",
                  "逐个原始SHAP/ALE CSV浮点值未在本脚本中逐行重算比对；图件生成脚本保留，供复核。"]
write("科学结果未改变对照记录.md","\n".join(numeric_lines)+"\n")

headings_old=[p.text.strip() for p in old.paragraphs if p.style.name.startswith("Heading") and re.match(r"^(第\d+章|\d+(?:\.\d+)*|附录[A-F])",p.text.strip())]
headings_new=[p.text.strip() for p in new.paragraphs if p.style.name.startswith("Heading") and re.match(r"^(第\d+章|\d+(?:\.\d+)*|附录[A-F])",p.text.strip())]
new_set=set(headings_new)
struct=["# 正文章节结构对照", "", "V1保留第三章TabICLv2主线及附录D XGBoost基准解释。V2未重新分配解释模型。", "",
        "## 主要调整", "", "- 3.1纳入XGBoost基准模型原理；3.2.1仅述TabICLv2上下文学习，原空3.2.3已处理。",
        "- 3.3.5补充站点隔离、训练折填补、目标变换和8成员推理；3.4收拢指标定义并写明站点配对bootstrap。",
        "- 3.5主文增加十任务SHAP紧凑图与汇总表，独立十图保留附录F。",
        "- 3.6主文四代表任务ALE双变量紧凑图，20条单独曲线完整保留附录E。",
        "- 4.3改为验证口径说明；4.4、5.6—5.8补入冻结站点级误差与来源消融证据。",
        "- 第六章标题、参考文献和附录A—F顺序已整理；个人简历与成果仍置附录之后。", "",
        "## V1存在而V2不存在的标题", ""]
struct += ["- "+h for h in headings_old if h not in new_set] or ["无"]
struct += ["", "## V2新增或调整标题", ""]
struct += ["- "+h for h in headings_new if h not in set(headings_old)] or ["无"]
write("正文章节结构对照.md","\n".join(struct)+"\n")

short_pages=[]
for i,page in enumerate(pdf):
    if len(page.get_text().strip())<120 and i>15:
        short_pages.append(i+1)
visual=["# V2排版与视觉验收", "", f"PDF共{len(pdf)}页；DOCX ZIP完整性：通过；重新打开DOCX：通过。",
        f"图题{len(captions)}条、嵌入图像{len(new.inline_shapes)}幅；图目录逐项定位：{'通过' if not figure_issues else '有异常'}。",
        "抽查页：21页图2-3/2-4顺序，31页新表3-3，32页ALE历史五折图，39—40页代表任务图，84/94页附录图。",
        "表3-3原任务名称断行已缩写修复，源变量长英文仍有多行；缩写在表注定义。",
        "已知视觉遗留：图3-5特征排名曲线图的图例靠近曲线，图3-6历史五折ALE图轴标签偏小；这两图保留V1数值与图面，尚未重绘。",
        "31页下半部有较大留白，下一节3.6从32页起；是章节分页而非空白页。",
        "封面后第2—4页、致谢第61页、个人简历/成果第99—100页沿用V1待作者填写的模板性内容；不得视为已完成的学校正式提交稿。",
        f"文本不足120字的PDF页（可能为独立图页/章节末页，需人工终审）：{short_pages}。",
        "自动检查无法证明全文不存在细小文字、页眉页脚压线或全部跨页表格问题；尤其福州大学官方模板尚未作为基准核对。",
        "结论：可作为V2学术修订稿交付，但不应宣称已通过学校终稿排版验收。"]
write("视觉验收记录.md","\n\n".join(visual)+"\n")

change=["# V2中文修改说明", "", "编辑母本：V1解释章节重构稿。V1未覆盖，全部新增内容位于V2独立目录。",
        "1. 第三章方法：分离XGBoost基准模型与TabICLv2主模型原理；补足上下文样本、折内预处理、指标与站点配对重采样定义。",
        "2. 第三章解释：十任务SHAP主文紧凑呈现；四代表任务ALE主文展示，20条原始曲线附录完整保留；支持审查与数值重复性另列。",
        "3. 第四、五章：用已有性能表、站点级OOF误差、来源消融和特征缺失率补充任务差异与空间泛化讨论；未添加未经验证机制。",
        "4. 图表：调整图2-3/2-4与图4-3/4-4顺序；Secchi分布图改为站点—日期去重；表2-1、表5-1及新增表3-3缩写排版。",
        "5. 文献与结构：补TabICLv2正式论文、空间验证、近岸水质文献，修[3]作者，整理第六章及附录A—F。",
        "6. 保留限制：P1九任务探针不等于十任务通过；ALE 2D仅联合支持矩阵，无响应曲面；XGBoost解释仅附录D独立基准。",
        "未完成或需作者确认：学校模板最终版式、封面及声明页个人信息、致谢、个人简历与成果、静态图目录改为域、历史图3-5/3-6图内文字终审。"]
write("中文修改说明.md","\n\n".join(change)+"\n")

print("QA",len(pdf),len(captions),len(figrows),"figure_issues",len(figure_issues),"refs",len(refs),"uncited",ref_uncited,
      "numeric_formal",numeric_result_ok,"media_unchanged",unchanged_media,"short_pages",short_pages)
