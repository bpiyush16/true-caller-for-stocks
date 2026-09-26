from __future__ import annotations

import shutil
from pathlib import Path

from docx import Document


SOURCE_DOCX = Path(r"D:\Downloads\Piyush_Bale_CS300_Report.docx")
OUTPUT_DOCX = Path(
    r"C:\Users\Piyush Bale\OneDrive\Desktop\Market Regim\Piyush_Bale_CS300_Report_manual_edited.docx"
)


CHANGES: list[tuple[int, str, str]] = [
    (69, "Tesla  T4  GPU  acceleration  demonstrate  that  the  Multi-Asset  variant  achieves  a", "Tesla  T4  GPU  acceleration  demonstrate  that  the  Multi-Asset  variant  achieves  a"),
    (70, "backtest  accuracy  of  97.53%  for  the  2025–2026  period,  representing  a  significant", "proxy-label  agreement  of  97.53%  for  the  2025?2026  period,  representing  a"),
    (71, "+10.80  percentage  point  improvement  over  models  restricted  to  S&P;  500  data", "+10.80  percentage  point  improvement  over  models  restricted  to  S&P;  500  data"),
    (72, "alone.  Furthermore,  the  Multi-Asset  model  exhibited  a  0.94  precision  score  for  the", "alone.  Furthermore,  the  Multi-Asset  model  exhibited  a  0.94  precision  score  for  the"),
    (73, "\"Crisis\"  regime,  successfully  flagging  the  March  2023  regional  banking  stress  as", "\"Crisis\"  proxy  regime,  while  the  March  2023  regional  banking  stress  episode  is"),
    (74, "\"Risk-Off\"  with  99.1%  confidence  while  maintaining  an  out-of-sample  profile.  This", "treated  as  a  qualitative  \"Risk-Off\"  case  study  rather  than  externally  validated"),
    (75, "research confirms that financial regimes are a systemic phenomenon best predicted", "market  ground  truth.  These  results  indicate  that  financial  regimes  are  better"),
    (76, "through the lens of inter-market relationships.", "studied through the lens of inter-market relationships within an internal framework."),
    (150, "Figure 5.1:  Rolling backtest: Actual vs. Predicted regimes, predicted class probabilities,", "Figure 5.1:  Rolling backtest: Proxy labels vs. Predicted regimes, predicted class probabilities,"),
    (158, "alongside rolling 30-day backtest accuracy and model confidence over the evaluation", "alongside rolling 30-day backtest agreement and model confidence over the evaluation"),
    (242, "(cid:127)  State Discovery Layer: A Gaussian HMM that performs unsupervised", "(cid:127)  State Discovery Layer: A Gaussian HMM that performs unsupervised"),
    (243, "clustering to define the boundaries between Calm, Risk-Off, and Crisis regimes.", "clustering to identify two broad latent market states for exploratory regime analysis."),
    (245, "(cid:127)  Predictive Logic Layer: A supervised sequence-to-label LSTM that generates", "(cid:127)  Predictive Logic Layer: A supervised sequence-to-label LSTM that generates"),
    (246, "\"tomorrow forecasts\" based on the patterns discovered in the historical latent", "\"tomorrow forecasts\" for custom proxy regimes using trailing engineered market"),
    (247, "space.", "features and the labeling logic defined for the supervised stage."),
    (250, "replays the model across unseen data (2025–2026) to quantify accuracy and", "replays the model across unseen data (2025?2026) to quantify agreement and"),
    (267, "accuracy gap between multi-asset and single-asset configurations.", "proxy-label agreement gap between multi-asset and single-asset configurations."),
    (415, "(cid:127)  Target Labeling: It utilizes a custom \"Soft-Labeler\" that blends HMM states", "(cid:127)  Target Labeling: It utilizes a custom \"Soft-Labeler\" based on"),
    (416, "with sigmoid-based volatility scores to produce smooth, non-flickering regime", "sigmoid-based volatility, return, and VIX heuristics with exponential"),
    (417, "labels.", "smoothing to produce smooth, non-flickering proxy regime labels."),
    (419, "(cid:127)  Classification Engine: A 2-layer LSTM with dropout is trained on these labels", "(cid:127)  Classification Engine: A 2-layer LSTM with dropout is trained on these"),
    (420, "to forecast the regime for the \"next-day\" based on the trailing 20 days of market", "proxy labels to forecast the regime for the \"next-day\" based on the trailing 20 days of market"),
    (421, "features.", "features."),
    (488, "(cid:127)  Normalization: Raw prices are converted to percentage changes (log-returns)", "(cid:127)  Normalization: Raw prices are converted to daily percentage returns"),
    (489, "and processed using a MinMaxScaler to bound values between [0, 1], which was", "and processed using a MinMaxScaler to bound values between [0, 1], which was"),
    (490, "found essential for the convergence of the LSTM Autoencoder.", "found essential for the convergence of the LSTM Autoencoder."),
    (521, "Stage 4 (Supervised Training): A supervised LSTM classifier is trained to forecast", "Stage 4 (Supervised Training): A supervised LSTM classifier is trained to forecast"),
    (522, "the  HMM-identified  regimes,  utilizing  Class  Weights  to  address  the  rarity  of  Crisis", "custom 3-class proxy regimes generated by the soft-labeling logic, utilizing Class"),
    (523, "events.", "Weights to address the rarity of Crisis events."),
    (525, "Stage 5 (Out-of-Sample Prediction): The model is run on the 2023 calendar year,", "Stage 5 (Out-of-Sample Prediction): The model is run on the 2023 calendar year,"),
    (526, "validating  its  ability  to  detect  the  March  2023  Bank  Crisis  with  99%  confidence", "serving as a proxy-label agreement check and as a qualitative case study during"),
    (527, "despite no prior exposure.", "the March 2023 regional banking stress episode."),
    (530, "performed from January 2025 to April 2026, quantifying the +10.8% accuracy gap in", "performed from January 2025 to April 2026, quantifying the +10.8% agreement gap in"),
    (534, "the 97.53% overall accuracy.", "the 97.53% overall agreement with the internal proxy labels."),
    (637, "During Stage 1, a race condition occurred within the yfinance multi-threading engine,", "During Stage 1, a race condition occurred within the yfinance multi-threading engine,"),
    (638, "leading  to  OperationalError:  database  is  locked.  This  was  identified  as  an  SQLite", "leading  to  cache-related  OperationalError  risk.  This  was  traced  to  Yahoo  Finance"),
    (639, "concurrency  conflict  where  multiple  threads  attempted  to  write  to  the  local  cache", "and SQLite cache contention where multiple requests could compete for local state"),
    (640, "simultaneously.  The  solution ", "simultaneously.  The  practical  mitigation  involved  hardening  the  download  stage"),
    (641, "ingestion", "with safer retry / timeout handling and more defensive data validation during"),
    (642, "(threads=False) to ensure thread-safe SQLite cache writes.", "ingestion rather than assuming the cache path was fully thread-safe."),
    (644, "forcing  sequential  data ", ""),
    (646, "involved ", ""),
    (666, "The  performance  of  the  market  regime  detection  engine  was  evaluated  using  a", "The  performance  of  the  market  regime  detection  engine  was  evaluated  using  a"),
    (667, "combination  of  classification  and  financial  backtesting  metrics  to  quantify  its", "combination  of  classification  and  financial  backtesting  metrics  to  quantify  its"),
    (668, "predictive accuracy and stability.", "agreement with internally generated proxy labels and overall stability."),
    (670, "(cid:127)  Total Accuracy: The percentage of trading days where the forecasted regime", "(cid:127)  Total Agreement: The percentage of trading days where the forecasted regime"),
    (671, "for \"tomorrow\" matched the actual ground-truth label.", "for \"tomorrow\" matched the internally generated proxy label."),
    (678, "(cid:127)  High-Confidence Accuracy (Acc", "(cid:127)  High-Confidence Agreement: The agreement achieved specifically on"),
    (679, "high", "days where the model's prediction probability exceeded a threshold of 0.80,"),
    (680, "days where the model's prediction probability exceeded a threshold of 0.80,", "indicating the reliability of the signal during periods of high certainty."),
    (683, "): The accuracy achieved specifically on", ""),
    (693, "The  primary  objective  was  to  evaluate  the  \"Multi-Asset  Advantage.\"  The  backtest", "The  primary  objective  was  to  evaluate  the  \"Multi-Asset  Advantage.\"  The  backtest"),
    (694, "was conducted over a period of 324 trading days, spanning from January 1, 2025, to", "was conducted over a period of 324 trading days, spanning from January 1, 2025, to"),
    (695, "April  21,  2026.  The  quantitative  results  demonstrate  a  clear  superiority  of  the", "April  21,  2026.  Within  the  internal  proxy-label  evaluation  framework,  the"),
    (696, "multi-asset approach.", "quantitative results demonstrate a clear superiority of the multi-asset approach."),
    (700, "Accuracy", "Agreement"),
    (704, "High-Conf Acc.", "High-Conf Agree."),
    (740, "The  Multi-Asset  model  exhibited  a  significant  accuracy  lead,  correctly  identifying", "The  Multi-Asset  model  exhibited  a  significant  agreement  lead,  correctly  matching"),
    (741, "nearly 11% more trading days than the S&P; 500-only model. This performance gap", "nearly 11% more proxy-labeled trading days than the S&P; 500-only model. This"),
    (742, "confirms  that  inter-market  signals—such  as  Gold  price  shifts  or  Bond  yield", "performance gap confirms that inter-market signals?such as Gold price shifts or Bond"),
    (743, "inversions—provide  critical  context  for  identifying  regime  boundaries  that  equity", "yield inversions?provide critical context that equity prices alone cannot capture."),
    (749, "Figure 5.1: Rolling backtest (January 2025 – April 2026): Actual vs. Predicted next-day", "Figure 5.1: Rolling backtest (January 2025 ? April 2026): Proxy labels vs. Predicted next-day"),
    (750, "regimes (top), predicted class probabilities for Calm, Risk-Off and Crisis (middle), and", "regimes (top), predicted class probabilities for Calm, Risk-Off and Crisis (middle), and"),
    (751, "prediction errors with model confidence overlay (bottom). Pink bands indicate misprediction", "prediction errors with model confidence overlay (bottom). Pink bands indicate misprediction"),
    (752, "days; grey vertical lines mark actual regime transitions.", "days; grey vertical lines mark proxy regime transitions."),
    (756, "The  following  tables  provide  a  granular  view  of  the  model's  ability  to  differentiate", "The  following  tables  provide  a  granular  view  of  the  model's  ability  to  differentiate"),
    (757, "between  the  three  market  states.  The  Multi-Asset  model  achieves  near-perfect", "between  the  three  proxy  market  states  used  in  the  evaluation  framework.  The"),
    (758, "scores  for  the  \"Calm  Market\"  and  maintains  very  high  precision  for  \"Risk-Off\"  and", "Multi-Asset  model  achieves  near-perfect  scores  for  the  \"Calm  Market\"  and  maintains"),
    (759, "\"Crisis.\"", "very high precision for \"Risk-Off\" and \"Crisis.\""),
    (930, "Figure 5.2: Regime comparison plot showing Multi-Asset vs. Single-Asset predicted regimes", "Figure 5.2: Regime comparison plot showing Multi-Asset vs. Single-Asset predicted regimes"),
    (931, "alongside the actual ground truth (top), rolling 30-day backtest accuracy for both variants", "alongside the proxy regime labels (top), rolling 30-day backtest agreement for both variants"),
    (932, "(middle), and model confidence over time with grey vertical lines marking actual regime", "(middle), and model confidence over time with grey vertical lines marking proxy regime"),
    (933, "transitions (bottom). The Single-Asset model's rolling accuracy collapses to below 40% during", "transitions (bottom). The Single-Asset model's rolling agreement collapses to below 40% during"),
    (934, "the mid-2025 crisis period.", "the mid-2025 crisis period."),
    (955, "Both  models  were  relatively  successful  in  identifying  the  \"Crisis\"  regime,  but  the", "Both  models  were  relatively  successful  in  identifying  the  \"Crisis\"  proxy  regime,  but  the"),
    (963, "the  Single-Asset  model  generated  only  290.  Furthermore,  the  accuracy  on  these", "the  Single-Asset  model  generated  only  290.  Furthermore,  the  agreement  on  these"),
    (964, "high-confidence days was significantly higher for the multi-asset variant (98.70% vs.", "high-confidence days was significantly higher for the multi-asset variant (98.70% vs."),
    (965, "89.66%). This indicates that the inter-market features not only improve accuracy but", "89.66%). This indicates that the inter-market features not only improve agreement but"),
    (971, "Despite its high overall accuracy, the Multi-Asset model only correctly predicted the", "Despite its high overall proxy-label agreement, the Multi-Asset model only correctly predicted the"),
    (998, "(cid:127)  Identification of the Multi-Asset Advantage: A comprehensive profiling of", "(cid:127)  Identification of the Multi-Asset Advantage: A comprehensive profiling of"),
    (999, "the performance gap between single-index and multi-asset models, establishing", "the performance gap between single-index and multi-asset models, establishing"),
    (1000, "that inter-market context leads to a +10.80% accuracy improvement.", "that inter-market context leads to a +10.80% proxy-label agreement improvement."),
    (1002, "(cid:127)  High-Precision Crisis Detection: The development of a model capable of", "(cid:127)  Illustrative Stress-Episode Signalling: A model that produced elevated"),
    (1003, "flagging systemic stress events, such as the March 2023 regional banking crisis,", "Risk-Off behaviour during the March 2023 regional banking stress episode,"),
    (1004, "with 99.1% confidence while maintaining a fully out-of-sample profile.", "presented as a qualitative case study rather than externally validated ground truth."),
    (1006, "(cid:127)  End-to-End Forecasting Engine: The delivery of a functional inference", "(cid:127)  End-to-End Forecasting Engine: The delivery of a functional inference"),
    (1007, "system that provides probabilistic regime forecasts for the subsequent trading", "system that provides probabilistic regime forecasts for the subsequent trading"),
    (1008, "session, achieving an overall backtest accuracy of 97.53% for the 2025–2026", "session, achieving an overall backtest agreement of 97.53% for the 2025?2026"),
    (1009, "period.", "period within the internal proxy-label framework."),
    (1043, "While  the  current  engine  achieves  high  predictive  accuracy,  several  directions  for", "While  the  current  engine  achieves  high  proxy-label  agreement,  several  directions  for"),
    (1044, "future enhancement remain:", "future enhancement remain:"),
    (1073, "The engine was benchmarked across different lookahead configurations, draft model", "The engine was benchmarked over a 324-day rolling backtest within an internal"),
    (1074, "sizes, and caching configurations using a 324-day rolling backtest. In the absence of", "proxy-label evaluation framework. In this setting, the multi-asset engine achieved"),
    (1075, "single-asset restrictions, the multi-asset engine achieves a peak accuracy of 97.53%,", "97.53% agreement, while the single-asset configuration trailed by 10.80 percentage"),
    (1076, "reaching  a  throughput  of  98.70%  on  high-confidence  prediction  days.  Contrary  to", "points. These results indicate that inter-market context improves proxy-regime"),
    (1077, "common practice, relying solely on equity index data does not consistently improve", "modelling even though the reported metrics should not be interpreted as externally"),
    (1078, "performance  in  the  regime  detection  setting.  Instead,  the  single-asset  configuration", "validated ground-truth regime accuracy."),
    (1079, "underperforms  relative  to  the  multi-asset  baseline  by  10.80  percentage  points,", "Instead, they show that combining multiple market channels yields a more informative"),
    (1080, "indicating that the overhead of missing inter-market signals outweighs any simplicity", "internal representation than relying solely on equity index data."),
    (1085, "By achieving a 97.53% accuracy and providing clear, high-confidence signals—such", "By achieving 97.53% agreement within its proxy-label framework and providing"),
    (1086, "as  the  current  Risk-Off  forecast  for  April  21,  2026,  with  98.9%  confidence—this", "clear, high-confidence signals, this project establishes a useful foundation for"),
    (1087, "project establishes a robust framework for modern quantitative research and market", "future quantitative research on multi-asset regime modelling, external validation,"),
    (1088, "surveillance.", "and more responsive transition detection."),
]


def replace_paragraph_text(paragraph, new_text: str) -> None:
    if paragraph.runs:
        paragraph.runs[0].text = new_text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(new_text)


def main() -> None:
    if not SOURCE_DOCX.exists():
        raise FileNotFoundError(f"Source DOCX not found: {SOURCE_DOCX}")

    shutil.copy2(SOURCE_DOCX, OUTPUT_DOCX)
    doc = Document(str(OUTPUT_DOCX))

    mismatches: list[str] = []
    for index, old_text, new_text in CHANGES:
        actual = doc.paragraphs[index].text
        if actual != old_text:
            mismatches.append(
                f"Paragraph {index} mismatch.\nEXPECTED: {old_text}\nACTUAL:   {actual}"
            )
            continue
        replace_paragraph_text(doc.paragraphs[index], new_text)

    if mismatches:
        raise ValueError("\n\n".join(mismatches))

    doc.save(str(OUTPUT_DOCX))

    verify_doc = Document(str(OUTPUT_DOCX))
    failed_verifications = []
    for index, _, new_text in CHANGES:
        if verify_doc.paragraphs[index].text != new_text:
            failed_verifications.append(
                f"Paragraph {index} failed verification.\n"
                f"EXPECTED: {new_text}\n"
                f"ACTUAL:   {verify_doc.paragraphs[index].text}"
            )

    if failed_verifications:
        raise ValueError("\n\n".join(failed_verifications))

    print(f"Updated DOCX written to: {OUTPUT_DOCX}")
    print(f"Applied changes: {len(CHANGES)} paragraphs")


if __name__ == "__main__":
    main()
