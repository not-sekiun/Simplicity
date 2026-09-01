# Demo video — evaluation results section

Spoken narration, ~2:20. Numbers are deliberately sparse: the tables carry the
detail, the voice carries the *argument*. Anything in `[ON SCREEN]` is a cue.

`[NAME]` = the product name once it's chosen.

---

### Beat 1 — why there are four tiers, not one number (~20s)

> `[ON SCREEN: the four-tier results table]`

"Most detectors get reported as one number on one benchmark. That number is
almost always measured on clean images, from generators the model already knows.

So we don't report one number. We report four tiers, and each one is chosen to
break a different assumption. Here they are — I'll come back to the ones that
matter."

---

### Beat 2 — clean vs transformed (~30s)

> `[ON SCREEN: robustness summary — clean / transformed / worst]`
> `[or: stats/charts/07_robustness_summary.png]`

"Every tier gets scored twice: clean, and again across seventeen transformations
— compression, blur, resize, noise, colour shifts, and chains of them together.

The important part isn't the scores. It's that **eleven of those eighteen views
were never trained on.** Harsher compression, heavier blur, heavier noise, and
all three chained pipelines are held out. And we use one fixed threshold for
every view — re-tuning per transform is the easiest way to make a fragile
detector look robust on a slide.

The weak point is the same everywhere: heavy noise. We're not hiding that. It's
twice the strongest noise the model ever saw in training."

---

### Beat 3 — the held-out generator (~35s)  ← **the money shot**

> `[ON SCREEN: DALL·E 3 row highlighted, then the per-view improvement]`

"Here's the test I actually care about.

We pulled three current image generators. We trained on two of them — Midjourney
v6 and nano-banana — and we held the third one back completely. DALL·E 3. The
model has never seen it, from a publisher it's never seen.

`[beat]`

Adding those two modern generators improved DALL·E 3 detection on **all eighteen
views.** Every single one. At a stricter threshold than the previous model used.

That's the difference between a detector that memorised some generators and one
that learned something transferable. And it's falsifiable — if it hadn't
generalised, this tier is exactly where it would have shown."

---

### Beat 4 — the era inversion (~25s)

> `[ON SCREEN: per-generator recall table, sorted]`

"Now look at where it actually fails.

Our worst generators are the **oldest** ones — research models from 2021 and
2022. Our best are the newest, including the one we held out.

That inversion is the answer to the obvious question: will this still work next
year? The evidence says the newer the generator, the better this does."

---

### Beat 5 — false positives, and the threshold (~30s)

> `[ON SCREEN: platform FPR table, then the threshold sweep chart]`

"One more, and it's the one that decides whether this is shippable.

The failure that kills this product isn't missing an AI image. It's telling
someone their own photograph is fake.

So the headline metric is false positives on **real** photographs — real Reddit,
X and Facebook photos, already compressed by those platforms. At our operating
point that's about two and a half percent, at ninety-eight percent recall.

That threshold isn't tuned on the numbers you're seeing. It's swept on one half
of the data and reported on the other. At the default of 0.5, false positives are
ten times worse — that trade is the entire product decision."

---

### Beat 6 — what we threw away (~20s)

> `[ON SCREEN: the rejected-datasets table]`

"Last thing, and it's the part I'd want you to judge us on.

We found a change that made our false-positive rate almost four times better. We
threw it away — because the real photos it trained on came from the same scrape
we score against. The number would have been great and meaningless.

Three other datasets got rejected the same way, including one that turned out to
be real photographs labelled as AI. None of them were caught by a metric. All of
them were caught by looking at the pixels."

---

## Notes for delivery

- **Pause after "all eighteen views."** That's the line the whole section is
  built around; let it land before moving on.
- **Don't read numbers off the tables.** Every figure spoken aloud above is
  rounded on purpose — "about two and a half percent", "almost four times
  better". Precision lives on screen.
- **Do not say "ten unseen generators."** The OOD tier has **four** truly unseen
  generators relative to the shipping model; the old "10" figure was measured
  against a smaller training pool. Say "a fully held-out generator" and point at
  DALL·E 3 instead — it's the stronger claim anyway.
- **If you're over time**, cut Beat 4 (era inversion). It's the most interesting
  and the least necessary — Beats 3 and 5 carry the argument alone.
- **If you have 20 seconds spare**, add after Beat 2: *"Because the backbone is
  frozen, all of this runs on a thousand and twenty-five trained parameters —
  one linear layer, on one consumer GPU."*

## Visual assets

| beat | asset |
|---|---|
| 1 | four-tier results table (README top) |
| 2 | `stats/charts/07_robustness_summary.png` |
| 3 | DALL·E 3 row + `stats/charts/03_robustness_per_view.png` |
| 4 | `stats/charts/05_generator_recall.png` |
| 5 | platform FPR table + `stats/charts/04_threshold_sweep.png` |
| 6 | rejected-datasets table (`TABLE_datasets.md`) |
