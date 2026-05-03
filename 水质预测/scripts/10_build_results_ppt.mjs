import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { dirname, resolve } from "node:path";

const artifactToolPath = pathToFileURL(
  "C:/Users/刘逸凡/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs"
).href;

const {
  Presentation,
  PresentationFile,
  column,
  row,
  grid,
  layers,
  panel,
  text,
  shape,
  chart,
  rule,
  fill,
  hug,
  fixed,
  wrap,
  grow,
  fr,
  auto,
} = await import(artifactToolPath);

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const OUT_DIR = resolve(ROOT, "reports", "ppt");
mkdirSync(OUT_DIR, { recursive: true });

function readCsv(path) {
  const raw = readFileSync(resolve(ROOT, path), "utf8").replace(/^\uFEFF/, "").trim();
  const lines = raw.split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(",");
    const obj = {};
    headers.forEach((h, i) => {
      const v = values[i];
      const num = Number(v);
      obj[h] = Number.isFinite(num) && v !== "" ? num : v;
    });
    return obj;
  });
}

const fusion = readCsv("reports/xgboost_surface_all_targets_fusion_metrics.csv");
const ablation = readCsv("reports/xgboost_surface_all_targets_ablation_common_metrics.csv");
const landsatQuality = readCsv("reports/landsat_quality_report.csv");

const colors = {
  ink: "#10212B",
  muted: "#58717C",
  faint: "#D8E1E5",
  bg: "#F7FAFA",
  blue: "#2F80ED",
  teal: "#00A6A6",
  green: "#3BA776",
  orange: "#F2994A",
  red: "#D95C5C",
  navy: "#14324A",
  white: "#FFFFFF",
};

const deck = Presentation.create({ slideSize: { width: 1920, height: 1080 } });

function addSlide(bg = colors.bg) {
  const slide = deck.slides.add();
  slide.background.fill = bg;
  return slide;
}

function titleBlock(title, subtitle = "") {
  return column({ name: "title-block", width: fill, height: hug, gap: 12 }, [
    text(title, {
      name: "slide-title",
      width: fill,
      height: hug,
      style: { fontSize: 54, bold: true, color: colors.ink, fontFace: "Aptos Display" },
    }),
    subtitle
      ? text(subtitle, {
          name: "slide-subtitle",
          width: wrap(1320),
          height: hug,
          style: { fontSize: 24, color: colors.muted, fontFace: "Aptos" },
        })
      : rule({ name: "title-rule", width: fixed(160), stroke: colors.teal, weight: 5 }),
  ]);
}

function smallLabel(value, color = colors.muted) {
  return text(value, {
    width: fill,
    height: hug,
    style: { fontSize: 18, color, fontFace: "Aptos" },
  });
}

function metric(value, label, accent = colors.teal) {
  return column({ width: fill, height: hug, gap: 8 }, [
    text(value, {
      width: fill,
      height: hug,
      style: { fontSize: 58, bold: true, color: accent, fontFace: "Aptos Display" },
    }),
    text(label, {
      width: fill,
      height: hug,
      style: { fontSize: 19, color: colors.muted, fontFace: "Aptos" },
    }),
  ]);
}

function note(textValue) {
  return text(textValue, {
    width: fill,
    height: hug,
    style: { fontSize: 17, color: colors.muted, fontFace: "Aptos" },
  });
}

function bullet(items) {
  return column(
    { width: fill, height: hug, gap: 14 },
    items.map((item) =>
      row({ width: fill, height: hug, gap: 14 }, [
        shape({ width: fixed(10), height: fixed(10), fill: colors.teal, borderRadius: "rounded-full" }),
        text(item, {
          width: fill,
          height: hug,
          style: { fontSize: 20, color: colors.ink, fontFace: "Aptos" },
        }),
      ])
    )
  );
}

function authoredTable(headers, rows, widths) {
  return column({ width: fill, height: hug, gap: 0 }, [
    grid(
      {
        width: fill,
        height: hug,
        columns: widths.map(fr),
        padding: { x: 0, y: 0 },
        columnGap: 0,
      },
      headers.map((h, i) =>
        text(h, {
          width: fill,
          height: fixed(34),
          style: { fontSize: 14, bold: true, color: colors.navy, fontFace: "Aptos" },
        })
      )
    ),
    rule({ width: fill, stroke: colors.faint, weight: 2 }),
    ...rows.map((r) =>
      grid(
        {
          width: fill,
          height: hug,
          columns: widths.map(fr),
          padding: { x: 0, y: 6 },
          columnGap: 0,
        },
        r.map((cell) =>
          text(String(cell), {
            width: fill,
            height: hug,
            style: { fontSize: 14, color: colors.ink, fontFace: "Aptos" },
          })
        )
      )
    ),
  ]);
}

function simpleLine(label, value, accent) {
  return row({ width: fill, height: hug, gap: 16 }, [
    text(label, {
      width: fixed(190),
      height: hug,
      style: { fontSize: 24, bold: true, color: colors.ink, fontFace: "Aptos" },
    }),
    shape({ width: fixed(Math.max(12, value)), height: fixed(18), fill: accent, borderRadius: "rounded-full" }),
    text(`${Math.round(value)}%`, {
      width: fixed(80),
      height: hug,
      style: { fontSize: 20, color: colors.muted, fontFace: "Aptos" },
    }),
  ]);
}

function addFooter(slide, page, source = "数据：EPD/HKO/CDS/GEE；模型：XGBoost；训练 2014-2022，测试 2023-2024") {
  slide.compose(
    row({ width: fill, height: hug, padding: { x: 76, y: 28 }, gap: 20 }, [
      text(source, { width: fill, height: hug, style: { fontSize: 13, color: "#7A8F98", fontFace: "Aptos" } }),
      text(String(page).padStart(2, "0"), { width: fixed(46), height: hug, style: { fontSize: 13, color: "#7A8F98", fontFace: "Aptos", bold: true } }),
    ]),
    { frame: { left: 0, top: 998, width: 1920, height: 82 }, baseUnit: 8 }
  );
}

// 1. Cover
{
  const slide = addSlide("#EAF4F4");
  slide.compose(
    grid({ width: fill, height: fill, columns: [fr(0.8), fr(1.2)] }, [
      panel({ width: fill, height: fill, fill: colors.navy, padding: { x: 86, y: 92 } },
        column({ width: fill, height: fill, gap: 30 }, [
          text("香港近岸水质预测", { width: fill, height: hug, style: { fontSize: 78, bold: true, color: colors.white, fontFace: "Aptos Display" } }),
          text("多源过程因子 + 夜光 + Landsat 遥感的最小可跑版本结果分析", {
            width: fill,
            height: hug,
            style: { fontSize: 30, color: "#D7EFF0", fontFace: "Aptos" },
          }),
          rule({ width: fixed(240), stroke: colors.teal, weight: 6 }),
          text("阶段成果汇报 | v0 pipeline", {
            width: fill,
            height: hug,
            style: { fontSize: 22, color: "#A8CDD0", fontFace: "Aptos" },
          }),
        ]),
      ),
      column({ name: "cover-metrics", width: fill, height: fill, gap: 44, padding: { x: 118, y: 150 } }, [
        metric("1038", "表层水质样本", colors.navy),
        metric("4", "Chl-a、SD、TN、TP 已跑通", colors.teal),
        metric("35", "最终模型表字段", colors.green),
        metric("F3", "融合模型形成可复现基线", colors.orange),
      ]),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
}

// 2. Scope
{
  const slide = addSlide();
  slide.compose(
    column({ width: fill, height: fill, padding: { x: 84, y: 72 }, gap: 48 }, [
      titleBlock("v0 的核心价值：先跑通主流程，再逐层扩展", "这一步把研究从“可行性设想”推进到“可复现实验基线”。"),
      grid({ width: fill, height: fill, columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 34 }, [
        metric("2014-2024", "时间范围", colors.navy),
        metric("8", "代表站点", colors.teal),
        metric("Surface", "第一阶段水层", colors.green),
        metric("XGBoost", "基线模型", colors.orange),
      ]),
      authoredTable(
        ["模块", "第一版处理", "当前状态"],
        [
          ["水质", "EPD 历史海水水质；保留表层 Chl-a/SD/TN/TP", "完成"],
          ["过程因子", "ERA5 气象、HKO 降水、HKO 潮汐", "完成"],
          ["人类活动代理", "VIIRS 月度夜光 avg_rad/cf_cvg", "完成"],
          ["遥感", "Landsat 8/9 月尺度波段、指数、质量标记", "完成并清洗"],
        ],
        [0.7, 1.7, 0.6]
      ),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
  addFooter(slide, 2);
}

// 3. Data processing
{
  const slide = addSlide();
  slide.compose(
    column({ width: fill, height: fill, padding: { x: 84, y: 72 }, gap: 34 }, [
      titleBlock("数据处理不是清洗杂活，而是模型有效性的前提", "每个数据源都被压到同一个站点-月份或采样日期尺度，保证能与 EPD 实测水质对齐。"),
      authoredTable(
        ["数据源", "做了什么处理", "为什么这样处理", "对结果的影响"],
        [
          ["EPD 水质", "筛选 2014-2024、8 站点、Surface Water；统一四个目标字段", "保留可验证实测基准，降低 v0 复杂度", "四个目标均可训练；TN/TP 样本略少"],
          ["HKO 降水", "构造采样日前 3 天、14 天和当月累计降水", "表达径流、营养盐输入和透明度扰动", "过程模型更稳定"],
          ["HKO 潮汐", "以日高潮-低潮构造 tidal_range", "无采样时刻时先不用复杂潮相", "提供水动力一阶近似"],
          ["ERA5", "小时气象聚合为月均 t2m/u10/v10/tcc", "先保证 v0 可跑通", "后续升级区域平均"],
          ["VIIRS", "站点缓冲区提取月均夜光和质量覆盖", "人类活动和岸线压力代理", "补充过程驱动信息"],
          ["Landsat", "补算指数、插值补缺，并保留质量标记", "近岸水体遥感缺失不可忽略", "样本增加，F3 开始优于 F1"],
        ],
        [0.62, 1.25, 1.15, 1.15]
      ),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
  addFooter(slide, 3);
}

// 4. Landsat quality
{
  const slide = addSlide();
  slide.compose(
    grid({ width: fill, height: fill, padding: { x: 84, y: 72 }, rows: [auto, fr(1)], columns: [fr(1.15), fr(0.85)], columnGap: 58, rowGap: 42 }, [
      titleBlock("Landsat 缺失处理改变了实验结论", "原始遥感缺失较多；补齐并显式标记质量后，融合模型从“被拖累”变成“有增益”。"),
      column({ width: fill, height: fill, gap: 30 }, [
        chart({
          name: "landsat-missing-chart",
          chartType: "bar",
          width: fill,
          height: fixed(420),
          config: {
            title: "Landsat 特征缺失率变化（%）",
            categories: ["原始", "插值后", "最终"],
            series: [{ name: "缺失率", values: [27.0, 0.2, 0.0] }],
          },
        }),
        note("处理策略：短缺口按站点时间序列线性插值；剩余缺口用站点-月份气候态、月份中位数和全局中位数兜底。"),
      ]),
      column({ width: fill, height: fill, gap: 30 }, [
        metric("280", "存在 Landsat 缺失并被标记的站点-月份", colors.orange),
        metric("22", "valid_count=0 或 no_data 的站点-月份", colors.red),
        metric("597→846", "Chl-a 训练样本数提升", colors.green),
        metric("-0.204→-0.017", "Chl-a F3 测试 R² 改善", colors.teal),
      ]),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
  addFooter(slide, 4);
}

// 5. Fusion performance
{
  const slide = addSlide();
  const categories = fusion.map((d) => d.target);
  slide.compose(
    grid({ width: fill, height: fill, padding: { x: 84, y: 72 }, rows: [auto, fr(1)], columns: [fr(1.1), fr(0.9)], columnGap: 54, rowGap: 40 }, [
      titleBlock("四个目标变量已形成统一基线", "SD/TN/TP 的融合模型已取得正 R²，Chl-a 仍是下一轮优化重点。"),
      chart({
        name: "fusion-r2-chart",
        chartType: "bar",
        width: fill,
        height: fill,
        config: {
          title: "融合模型 F3：测试集 R² / NSE",
          categories,
          series: [{ name: "R²", values: fusion.map((d) => Number(d.r2.toFixed(3))) }],
        },
      }),
      authoredTable(
        ["目标", "Train", "Test", "RMSE", "MAE", "R²"],
        fusion.map((d) => [d.target, d.train_rows, d.test_rows, d.rmse.toFixed(3), d.mae.toFixed(3), d.r2.toFixed(3)]),
        [0.7, 0.7, 0.7, 0.8, 0.8, 0.7]
      ),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
  addFooter(slide, 5);
}

// 6. Ablation
{
  const slide = addSlide();
  const targets = ["Chl_a", "SD", "TN", "TP"];
  const series = ["F1_process", "F2_remote_sensing", "F3_fusion"].map((model) => ({
    name: model.replace("F1_process", "F1 过程").replace("F2_remote_sensing", "F2 遥感").replace("F3_fusion", "F3 融合"),
    values: targets.map((t) => Number(ablation.find((d) => d.target === t && d.model === model).r2.toFixed(3))),
  }));
  slide.compose(
    column({ width: fill, height: fill, padding: { x: 84, y: 72 }, gap: 38 }, [
      titleBlock("消融实验回答核心问题：融合是否有增益", "同一批共同样本下比较 F1/F2/F3，避免因 Landsat 缺失导致样本不一致。"),
      chart({
        name: "ablation-r2-chart",
        chartType: "bar",
        width: fill,
        height: fixed(520),
        config: {
          title: "共同样本消融：测试集 R²",
          categories: targets,
          series,
        },
      }),
      row({ width: fill, height: hug, gap: 42 }, [
        text("发现 1：Chl-a 与 SD 由融合模型 F3 最优，说明遥感补齐后提供增益。", { width: fill, height: hug, style: { fontSize: 25, bold: true, color: colors.ink, fontFace: "Aptos" } }),
        text("发现 2：TN/TP 的 F2 遥感模型 R² 略高，过程特征对营养盐的当前表达还偏弱。", { width: fill, height: hug, style: { fontSize: 25, bold: true, color: colors.ink, fontFace: "Aptos" } }),
      ]),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
  addFooter(slide, 6);
}

// 7. Interpretation
{
  const slide = addSlide();
  slide.compose(
    column({ width: fill, height: fill, padding: { x: 84, y: 72 }, gap: 42 }, [
      titleBlock("结果解释：模型表现与水质变量机制基本一致", "不同目标变量对遥感、过程因子和缺失处理的敏感性不同。"),
      grid({ width: fill, height: fill, columns: [fr(1), fr(1)], rows: [fr(1), fr(1)], columnGap: 44, rowGap: 34 }, [
        panel({ width: fill, height: fill, padding: { x: 28, y: 24 }, fill: colors.white, border: colors.faint }, bullet([
          "Chl-a：当前 R² 接近 0，说明藻类响应可能需要更精细的滞后、温盐、潮相或站点差异建模。",
          "融合模型仍优于 F1/F2，说明遥感和过程因子方向正确。"
        ])),
        panel({ width: fill, height: fill, padding: { x: 28, y: 24 }, fill: colors.white, border: colors.faint }, bullet([
          "SD：F3 最优，符合透明度同时受悬浮物、气象扰动和遥感水色影响的机制。",
          "这是当前最稳定、最适合作为论文主线展示的变量。"
        ])),
        panel({ width: fill, height: fill, padding: { x: 28, y: 24 }, fill: colors.white, border: colors.faint }, bullet([
          "TN：F2 的 R² 略高，说明站点空间差异和岸线压力可能被遥感/夜光间接捕捉。",
          "过程模型偏弱，后续需要加入径流、排污口或空间邻近特征。"
        ])),
        panel({ width: fill, height: fill, padding: { x: 28, y: 24 }, fill: colors.white, border: colors.faint }, bullet([
          "TP：与 TN 类似，遥感模型略优，但 F3 的 MAE 更低。",
          "说明融合模型对平均误差更稳，但极值解释仍需改进。"
        ])),
      ]),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
  addFooter(slide, 7);
}

// 8. Meaning
{
  const slide = addSlide("#F2F8F7");
  slide.compose(
    grid({ width: fill, height: fill, padding: { x: 84, y: 72 }, rows: [auto, fr(1)], columns: [fr(0.9), fr(1.1)], columnGap: 60, rowGap: 40 }, [
      titleBlock("实验意义：已经证明这条技术路线值得继续做", "v0 的价值不是追求最优分数，而是把数据、特征、模型和消融证据连成闭环。"),
      column({ width: fill, height: fill, gap: 26 }, [
        metric("可复现", "所有数据处理和建模步骤已脚本化", colors.navy),
        metric("可解释", "F1/F2/F3 区分过程驱动与遥感增益", colors.teal),
        metric("可扩展", "目标变量、水层、站点和传感器都能逐步增加", colors.green),
      ]),
      column({ width: fill, height: fill, gap: 22 }, [
        text("现在能支撑的论文论点", { width: fill, height: hug, style: { fontSize: 36, bold: true, color: colors.ink, fontFace: "Aptos Display" } }),
        bullet([
          "多源融合框架可用于香港近岸表层水质预测，尤其对透明度、TN、TP 已表现出正向预测能力。",
          "遥感特征质量控制会直接改变消融结论，是近岸水体建模不可跳过的环节。",
          "过程因子与遥感因子具有互补性，但互补强度取决于目标变量和遥感缺失处理质量。",
        ]),
      ]),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
  addFooter(slide, 8);
}

// 9. Next
{
  const slide = addSlide();
  slide.compose(
    column({ width: fill, height: fill, padding: { x: 84, y: 72 }, gap: 46 }, [
      titleBlock("下一步安排：先稳住 v0，再扩展复杂度", "建议按风险从低到高推进，避免一次性加入过多变量导致不可解释。"),
      grid({ width: fill, height: hug, columns: [fr(1), fr(1), fr(1)], columnGap: 42 }, [
        column({ width: fill, height: hug, gap: 18 }, [
          text("1. 稳定数据与模型", { width: fill, height: hug, style: { fontSize: 34, bold: true, color: colors.navy, fontFace: "Aptos Display" } }),
          bullet(["异常值和单位复核", "加入站点/分区特征", "比较 RF / LightGBM"]),
        ]),
        column({ width: fill, height: hug, gap: 18 }, [
          text("2. 强化遥感特征", { width: fill, height: hug, style: { fontSize: 34, bold: true, color: colors.teal, fontFace: "Aptos Display" } }),
          bullet(["扩大月窗口或季度平滑", "加入水体掩膜质量指标", "跑 Sentinel-2 增强实验"]),
        ]),
        column({ width: fill, height: hug, gap: 18 }, [
          text("3. 扩展科学问题", { width: fill, height: hug, style: { fontSize: 34, bold: true, color: colors.green, fontFace: "Aptos Display" } }),
          bullet(["扩展中层和底层", "空间分区验证", "稳定模型后做 SHAP"]),
        ]),
      ]),
      rule({ width: fill, stroke: colors.faint, weight: 2 }),
      text("优先级建议：先以 SD/TN/TP 建立稳定结果，再集中攻克 Chl-a；Chl-a 更适合作为后续优化重点，而不是当前唯一成败标准。", {
        width: fill,
        height: hug,
        style: { fontSize: 31, bold: true, color: colors.ink, fontFace: "Aptos" },
      }),
    ]),
    { frame: { left: 0, top: 0, width: 1920, height: 1080 }, baseUnit: 8 }
  );
  addFooter(slide, 9);
}

const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(resolve(OUT_DIR, "water_quality_prediction_results_v0.pptx"));

for (const [idx, slide] of deck.slides.items.entries()) {
  const png = await slide.export({ format: "png", width: 1920, height: 1080 });
  writeFileSync(resolve(OUT_DIR, `slide_${String(idx + 1).padStart(2, "0")}.png`), Buffer.from(await png.arrayBuffer()));
}

console.log(`Wrote ${resolve(OUT_DIR, "water_quality_prediction_results_v0.pptx")}`);
console.log(`Rendered ${deck.slides.items.length} slide previews to ${OUT_DIR}`);
