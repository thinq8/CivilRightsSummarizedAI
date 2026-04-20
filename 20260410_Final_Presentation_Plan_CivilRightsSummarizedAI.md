# Final Presentation Plan — Civil Rights Summarized AI

**Team:** Ink & Order — Liam Sandy, Nick Sleeper, Ishmail Khan, Jayant Sahai
**Course:** CMSE 495 Capstone, Michigan State University
**Community Partner:** Civil Rights Litigation Clearinghouse (University of Michigan Law School)
**Date:** 2026-04-10

> Companion file: `20260410_Final_Presentation_Plan_CivilRightsSummarizedAI.pptx`
> The PPTX contains the full storyboard for the final video (slides 1–5) and a visual version of this plan (slides 6–10).

---

## 1. Final Video — Storyboard Summary

Target length: ~3 minutes. Structure: Problem → Approach → Results → Impact.

### Act I — Setup & Problem (≈0:45)
1. **Cold Open** — Title card fades in over MSU green. Hook: "Ten thousand civil rights cases. One team of researchers. Every summary, written by hand."
2. **The Clearinghouse** — B-roll of the Clearinghouse website and a case summary page. Voiceover introduces the partner.
3. **The Ask** — Split screen: stack of PDFs → hand typing. "Meticulous. Does not scale."
4. **Our Mission** — Title card: "Can an AI help — without getting it wrong?"
5. **Meet the Team** — Four team photos with names and roles.
6. **The Real Problem** — Animated document stack growing past a "32K" context line.

### Act II — Build & Demo (≈1:30)
7. **Pipeline Diagram** — Ingest → Prepare → Fine-tune → Evaluate, animated flow.
8. **Demo: Ingestion** — Terminal screencast of `ingest-mock` pulling cases.
9. **Fragmentation Fix** — Before/after: 27K → 4.5K tokens via extract-first.
10. **Demo: Training** — Training loss curve drops from ~5,000 to ~1.2.
11. **Demo: The Model Speaks** — Raw case docs → generated summary paragraph.
12. **Evaluation Dashboard** — Bar chart: ROUGE, BERTScore, LLM-as-Judge.

### Act III — Results & Thanks (≈0:45)
13. **Key Result #1** — Figure 2: fragmentation histogram with its long tail.
14. **Key Result #2** — Figure 3: ROUGE + BERTScore vs. baselines.
15. **Key Result #3** — Figure 4: LLM-as-Judge scores on 5 dimensions.
16. **The Handoff** — Shot of the GitHub repo → Clearinghouse logo.
17. **Outro / Thank You** — "Thank you, Clearinghouse." GitHub URL on screen.

---

## 2. Symposium Lightning Talk — 5 Minute Plan

### Time Budget

| Time        | Beat              | Lead    | Key Line                                                                                            |
|-------------|-------------------|---------|-----------------------------------------------------------------------------------------------------|
| 0:00 – 0:30 | Hook & Team       | Liam    | Name the problem in one sentence. Introduce the team.                                               |
| 0:30 – 1:15 | Problem           | Ishmail | Why Clearinghouse staff need help. Scale + quality stakes.                                          |
| 1:15 – 2:30 | What we built     | Nick    | Pipeline diagram. Fragmentation fix. Short screenshot of model output.                              |
| 2:30 – 3:30 | Results           | Jayant  | ROUGE / BERTScore / LLM-as-Judge numbers with one clean chart.                                      |
| 3:30 – 4:15 | Impact & Handoff  | Liam    | What the Clearinghouse gets. Reproducibility. Ethics guardrails.                                    |
| 4:15 – 5:00 | Thanks & CTA      | All     | Thank the partner. Invite questions. Point at the poster / QR code.                                 |

### Script (memorize the first and last sentence of each block)

**LIAM — Hook (0:30)**
> "Civil rights cases matter. But every one of them gets summarized by hand — and that doesn't scale. We are Team Ink & Order, and this semester we built an AI that helps the Civil Rights Litigation Clearinghouse keep up."

**ISHMAIL — Problem (0:45)**
> "The Clearinghouse has catalogued more than ten thousand federal civil rights cases since 2002. Their staff writes every summary by hand, and a single case can include hundreds of court filings. We were asked: can a language model help — without losing the precision the law requires?"

**NICK — Build (1:15)**
> "We built a four-stage pipeline: ingest, prepare, fine-tune, evaluate. The tricky part was document fragmentation — cases blow past any context window. Our extract-first strategy compresses a 27K-token case down to about 4.5K tokens of structured facts, and then we LoRA fine-tuned Qwen2.5-7B on nearly ten thousand examples."

**JAYANT — Results (1:00)**
> "We evaluated three ways: ROUGE for lexical overlap, BERTScore for semantic similarity, and Claude as an LLM judge scoring legal reasoning and factual accuracy. Our fine-tuned model holds its own against strong baselines on the dimensions the Clearinghouse cares about most."

**LIAM — Impact & Thanks (0:45)**
> "Everything is reproducible and already in the Clearinghouse's hands — code, fixtures, and a notebook that regenerates every figure. Thank you to the Clearinghouse team at U-M Law, and thank you to CMSE 495 for the runway. We would love to talk — come find us after."

---

## 3. Team Roles

| Member        | Role on stage         | Responsibility                                                                                     |
|---------------|-----------------------|-----------------------------------------------------------------------------------------------------|
| Liam Sandy    | Lead presenter        | Opens and closes the talk. Owns the pipeline diagram slide and the handoff message.                |
| Nick Sleeper  | Build narrator        | Walks through the ingestion and training pipeline. Runs any live terminal demo.                    |
| Ishmail Khan  | Problem framer        | Sets the stakes for the Clearinghouse. Handles audience questions about the data.                  |
| Jayant Sahai  | Results lead          | Owns the evaluation chart. Fields questions about ROUGE, BERTScore, and the LLM judge.             |
| All four      | Networking            | After the talk, each person takes one corner of the poster and rotates with visitors.              |

---

## 4. Dress Code

Business casual, MSU colors where possible.
- Dark trousers or slacks (no jeans)
- Solid white or MSU-green collared shirt / blouse
- Closed-toe shoes — we will be standing the entire session
- Name tags worn on the right side so handshakes read them first

---

## 5. Day-Of Logistics

- Arrive 45 minutes early; test the HDMI and clicker
- Liam brings the laptop + a USB stick backup of the slides
- Nick brings a printed QR code linking to the GitHub repo
- Ishmail carries printed one-pagers for the partner and recruiters
- Jayant keeps the 5-minute timer running during the talk
- Rehearse twice end-to-end the night before

---

## 6. Anticipated Questions and Answers

**Q · Will this replace Clearinghouse staff?**
A · No. It is an assistive tool. Every generated summary still goes through human review before publication — that is a hard requirement we agreed to up front.

**Q · How do you know the model is not hallucinating?**
A · Our LLM-as-Judge scores factual accuracy as its own dimension, and we score on ROUGE for lexical overlap. Low scores on either flag a summary for closer review.

**Q · Why Qwen2.5-7B and not GPT-4 / Claude?**
A · The Clearinghouse needs something they can run privately on case data. A 7B open-weights model with LoRA adapters is deployable on a single GPU and keeps the data on their infrastructure.

**Q · What was the hardest part?**
A · Document fragmentation. Our first training run failed — loss spiked past 5,000 and eval went to NaN. Fixing it meant rebuilding the data prep pipeline around structured fact extraction.

**Q · Is the data public? Any privacy concerns?**
A · Yes — all source documents are public court filings. The Clearinghouse reference summaries are their IP, so we keep them out of the repo. We only process information already on the public record.

**Q · What would you do with another semester?**
A · Ship a small web UI for staff, test on held-out cases the model has never seen, and explore larger base models once compute allows.

---

## 7. Responsiveness to Prior Feedback

| Feedback                                                      | Our response                                                                                              | Where to see it |
|---------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|-----------------|
| Earlier drafts buried the problem statement.                  | Video now opens on the Clearinghouse and states the stakes in the first 15 seconds.                      | Frames 01–04    |
| Peer team asked for a clearer pipeline diagram.               | Added a dedicated pipeline frame (07) with the four stages labelled and animated.                        | Frame 07        |
| Reviewers wanted evidence, not just claims.                   | Added three results frames pulling directly from Figures 2–4 in the report.                              | Frames 13–15    |
| Previous plan had no time budget for the lightning talk.      | Built a per-beat time budget with a named lead for every 30–75 second slice.                             | Section 2       |
| Partner asked how we'd handle Q&A on accuracy.                | Pre-wrote answers for the six most likely questions — especially the hallucination one.                  | Section 6       |
| Instructor asked for a clearer handoff moment.                | Video now ends on an explicit "thank you" frame with the GitHub URL on screen.                           | Frame 17        |

---

## 8. Deliverable Checklist

- [x] Storyboard for the final video (PPTX, slides 1–5)
- [x] Presentation plan with script, roles, attire, logistics, Q&A (this file + PPTX slides 6–10)
- [x] File naming follows `YYYYMMDD_Final_Presentation_Plan_*`
- [x] Both files placed in the team's public project folder
- [ ] Rehearse the lightning talk twice before symposium day
- [ ] Upload the final exported video to the team drive 24 hours before the deadline
