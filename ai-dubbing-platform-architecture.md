# AI Dubbing Platform: A Greenfield Architecture Design

A first-principles design for a system that translates and dubs video or audio into other languages while preserving speaker identity, delivery, emotion, timing, and the overall viewing experience.

This document is deliberately opinionated and structurally honest about where the hard limits are. Several of the subsystems described below sit at the research frontier, and the design calls those out rather than papering over them.

---

## 0. The one constraint that shapes everything

Before any architecture, there is a single fact that determines most design decisions:

**Translation changes the duration and rhythm of speech, but the timeline is fixed.**

A line that takes 3.0 seconds in English may take 4.2 seconds as a faithful Spanish translation and 2.4 seconds in Japanese. Yet the video frame where the actor's mouth closes does not move, the cut to the next shot does not move, and the music swell underneath does not move. Every other quality goal (identity, emotion, prosody, lip-plausibility, naturalness) has to be achieved *while* satisfying a timing constraint that the source language defined and the target language resists.

This means the naive mental model, a clean linear pipeline of recognize, translate, synthesize, is wrong in one specific and important place. Translation, prosody, and timing are mutually constraining and must be solved jointly, not as independent sequential stages. Almost everything distinctive in this design follows from taking that seriously.

A second framing point: dubbing is not one problem. It is a family of problems along a frontier of cost, faithfulness, and naturalness. A YouTube explainer dubbed for reach has very different tolerances than a feature film dubbed for theatrical release. A world-class platform does not hardcode a single point on that frontier; it exposes the frontier as configuration (quality tiers and faithfulness knobs) and is honest with the user about the tradeoffs at each setting.

---

## 1. System architecture at a glance

The system is organized into five layers. Higher layers depend only on the interfaces of lower layers, not their internals.

1. **Surface layer**: API, CLI, and (future) UI. All three are thin clients over the same core. None of them contain dubbing logic.
2. **Orchestration layer**: a directed-acyclic-graph (DAG) execution engine that turns a project into a graph of stages and artifacts, schedules work, handles retries, and enforces incremental/resumable execution through content-addressed caching.
3. **Processing layer**: the dubbing stages themselves (separation, diarization, ASR, translation, synthesis, mixing, and so on), each implemented as an idempotent, independently scalable worker.
4. **Domain layer**: the canonical data model. The center of gravity here is a single editable document, the **Timeline**, which is the source of truth for the project and the substrate that every human edit and every render reads from and writes to.
5. **Storage layer**: a content-addressed object store for large media artifacts, a metadata database for the Timeline and project state, a vector store for speaker embeddings and translation memory, and an append-only event log for processing history and auditability.

The single most important architectural decision is that **the Timeline document is the spine**. Stages do not pass data to each other through ad-hoc tuples; they read from and write to a structured, versioned, content-addressed representation of the project. Human edits mutate the same representation. Renders are derived from it. This is what makes human-in-the-loop editing, resumability, incremental re-rendering, and auditability all fall out of one mechanism instead of being bolted on three times.

---

## 2. The Timeline: the canonical data model

Conceptually the Timeline resembles an edit decision list from video editing, extended for dubbing. It is an ordered set of **segments**, where a segment is the atomic unit of dubbing work (roughly a phrase or sentence, the granularity at which translation, timing, and synthesis are coupled).

Each segment carries, at minimum:

- **Source span**: start and end time in the original media, plus the source audio reference.
- **Speaker ID**: a stable identifier into the project's speaker registry.
- **Source text** with word-level timestamps and per-word ASR confidence.
- **Target text** (per target language), plus translation metadata (model/version, glossary hits, length-compliance score).
- **Prosody/affect parameters**: a compact representation of how the line was delivered (discussed in §6).
- **Timing plan**: the assigned target start and duration, the chosen speaking-rate factor, and the pause budget.
- **Voice profile reference**: which speaker voice and reference set to synthesize with.
- **Synthesis parameters**: the conditioning actually used to render this segment.
- **Render artifact reference**: a content hash pointing to the synthesized audio for this segment.
- **Review state**: machine confidence, human-edit flags, lock status, and approval status.

The Timeline is the metadata architecture (§15) made concrete. Three properties matter:

- It is **versioned**. Every edit produces a new immutable version; the project tracks a head pointer. This gives undo, branching (try two translations of a scene), and a clean answer to "what changed and who changed it."
- It is **content-addressed at the segment level**. The cache key for a segment's render is a hash of its synthesis-relevant fields (target text, prosody params, voice profile, timing plan, model versions). Change the translation of one line, and only that line's downstream artifacts invalidate. This is the foundation of incremental processing (§11).
- It is **the API**. The surface layer's primary resource is the Timeline. Editing a translation in the UI, patching a segment via the API, and a CLI batch edit are the same operation against the same document.

---

## 3. Pipeline stages and execution flow

The processing layer is a DAG, not a strict line, because some stages fan out, some are optional, and some feed back. The canonical flow:

```
                    +-- music stem ----------------------------+
                    |                                           |
ingest -> demux -> source -> +-- effects/ambient stem ---------+--> remix/master -> mux -> output
                  separation |                                  |        ^
                    |        +-- speech stem --+                |        |
                    |                          v                |        |
                    +-- video frames           diarization      |        |
                            |                   |               |        |
                            |                   v               |        |
                            |              speaker profiling     |        |
                            |              + reference select    |        |
                            |                   |               |        |
                            |                   v               |        |
                            +--> ASR --> linguistic analysis     |        |
                                          |                      |        |
                                          v                      |        |
                                    translation <----+           |        |
                                          |          |           |        |
                                          v          | (feedback)|        |
                                  prosody/affect      |          |        |
                                   analysis           |          |        |
                                          |           |          |        |
                                          v           |          |        |
                                   timing/duration ---+          |        |
                                   planner (joint optimizer)     |        |
                                          |                      |        |
                                          v                      |        |
                                   voice synthesis               |        |
                                          |                      |        |
                                          v                      |        |
                                   speech post-processing -------+        |
                                          |                               |
                                          +--- (optional) visual dubbing -+
```

Note the feedback arrow from the timing planner back to translation. That loop is the joint optimization that §0 demands, and it is described in §3.9.

The following subsections treat each major subsystem through six lenses: purpose, problems solved, data consumed, data produced, interactions, and tradeoffs. I have grouped tightly coupled stages and written some of this as prose rather than mechanically repeating a six-line template thirty times, but every major subsystem is covered on all six dimensions.

### 3.1 Ingestion and demux

**Purpose.** Accept arbitrary source media and normalize it into a known internal form.

**Problems solved.** Real input is heterogeneous: container formats, codecs, sample rates, frame rates, variable loudness, multiple existing audio tracks, embedded subtitles. Downstream stages should never see that variety.

**Consumes.** A source file or stream, plus project configuration (target languages, quality tier).

**Produces.** A normalized video essence (frames at a known rate and color space), a normalized audio essence (a known sample rate and bit depth), a probe report (duration, streams, loudness, detected channels), and any embedded subtitle/caption tracks captured as a reference artifact.

**Interactions.** Feeds demuxed audio to source separation and demuxed frames to both source separation's helper (active-speaker detection) and the eventual mux. Writes the probe report into project metadata.

**Tradeoffs.** Transcoding everything to a canonical intermediate costs storage and a quality generation, but the alternative (every stage handling every codec) is far worse for reliability. Where the source ships pre-separated stems (some studio content does), the system should accept "bring your own stems" and skip separation entirely, because original stems beat any separation model.

### 3.2 Source separation (stem extraction)

**Purpose.** Split the original audio into stems: **dialogue/speech**, **music**, **effects**, and **ambient/room tone**. The new translated dialogue will replace only the speech stem; everything else is preserved and re-mixed underneath.

**Problems solved.** You must not re-translate music or sound effects, and you must keep the original sonic world intact. You also need a clean speech stem both to recognize accurately and to extract uncontaminated voice references for cloning. This stage is therefore upstream of two very different consumers (recognition and identity), and its failures propagate to both.

**Consumes.** Normalized audio (and optionally video frames, because visual cues can assist separation in heavily mixed scenes).

**Produces.** Four (or more) stem artifacts plus a per-stem confidence/quality map over time, flagging regions where separation is uncertain (typically heavily scored or crowded scenes).

**Interactions.** Speech stem feeds diarization, ASR, and reference extraction. Music, effects, and ambient stems flow forward, untouched, to the remix stage. The quality map feeds reference selection (avoid low-confidence regions when picking voice samples) and QC.

**Tradeoffs.** This is one of the largest cumulative-error sources in the whole system. Perfect separation does not exist; energy bleeds between stems. The honest mitigations: use the cleanest regions for the highest-stakes purposes (reference extraction in particular), preserve the original full mix as a fallback for regions where the dialogue is sparse and separation buys little, and treat "bring your own stems" as a first-class path rather than an afterthought. Heavily-scored action scenes will always be the hardest, and the design should surface that to the user rather than silently degrading.

### 3.3 Speaker diarization and multi-speaker handling

**Purpose.** Determine *who spoke when*, and segment the speech stem into speaker-attributed turns that remain consistent across the entire piece of content.

**Problems solved.** Multi-speaker content requires assigning each line to the right voice. Mistakes here are extremely audible: a line rendered in the wrong character's voice is a glaring error even when everything else is perfect. The hard cases are overlapping speech, short back-and-forth turns, and speakers with similar voices.

**Consumes.** The speech stem and, critically, the video frames.

**Produces.** A speaker-segmented timeline (turn boundaries with provisional speaker labels) and the initial speaker registry for the project.

**Interactions.** Defines the segment skeleton that ASR, translation, and synthesis all hang off. Feeds the speaker registry that profiling refines.

**Tradeoffs and the key lever.** Audio-only diarization is the industry default and it is fragile precisely where dubbing cares most (overlap, similar voices, rapid dialogue). The major design lever is **multimodal diarization**: use **active-speaker detection from the video** (which on-screen face's lip motion correlates with the speech) to disambiguate. Video is a strong signal that most audio-only pipelines throw away. Fusing lip-motion-based active-speaker detection with audio embeddings and with ASR turn cues (speaker changes often align with sentence boundaries) produces far more robust attribution. The cost is complexity and a dependency on usable video; for audio-only inputs (podcasts) the system degrades to audio-plus-ASR diarization and should flag lower confidence on ambiguous turns for human review.

### 3.4 Speaker profiling, voice representation, and reference selection

**Purpose.** Build a stable, high-quality voice representation for each speaker, and select the specific reference audio that voice cloning will be conditioned on.

**Problems solved.** Cloning quality is bounded by reference quality. The same actor may have hours of audio across a film, but most of it is mixed with music, overlapped, whispered, shouted, or otherwise unrepresentative. The subsystem must (a) consolidate turns into consistent identities across the whole content (and, for series, across episodes), (b) compute a voice representation, and (c) curate the best reference set per speaker.

**Consumes.** Diarized speech segments, the separation quality map, and the per-word ASR confidence.

**Produces.** For each speaker: a **voice profile** (a speaker embedding, plus, where data permits, an optionally fine-tuned speaker-specific synthesis adapter), a curated **reference set** of clean, representative clips spanning the speaker's expressive range, and a **reference quality score** with an explicit "no adequate reference" failure state.

**Interactions.** The voice profile and reference set are consumed by synthesis. The speaker registry is the anchor for long-form and franchise consistency (§7). Reference quality feeds QC and the human-in-the-loop flow (a speaker with no clean reference is routed to a human to either approve a fallback voice or supply better audio).

**Voice representation, in detail.** A good representation is *disentangled*: it should capture speaker identity (timbre, vocal-tract characteristics) while factoring out linguistic content and, importantly, source-language accent. If the embedding entangles "this person speaking English," cloning into German inherits an English accent and the illusion breaks. The design therefore favors representations trained to separate identity from language and content, which is also what makes cross-lingual transfer (§5) tractable.

**Reference selection as its own problem.** This is effectively a quality-evaluation subsystem. Candidate clips are scored on cleanliness (from the separation quality map), absence of overlap, signal-to-noise, prosodic representativeness (you want neutral and expressive samples, not only shouting), and duration. The selector aims for coverage of the speaker's range, not just the single cleanest clip, because a one-note reference yields a one-note clone. The explicit failure state matters: when no clip clears the bar, the honest behavior is to flag it, not to clone from garbage.

**Tradeoffs.** Zero-shot cloning from a short reference is fast and scales to content with many minor speakers. Fine-tuning a speaker-specific model gives better identity and expressiveness but needs substantial clean data and compute, and only pays off for principal speakers with lots of screen time. The design should choose per speaker: fine-tune the leads, zero-shot the bit parts. That per-speaker decision is itself a configurable quality/cost tradeoff.

### 3.5 Speech recognition (ASR)

**Purpose.** Transcribe the source speech into text with word-level timing and confidence.

**Problems solved.** Everything textual downstream depends on this. Errors here propagate into translation (you translate the wrong words) and timing (word timestamps anchor alignment). The hard cases are domain vocabulary, names, code-switching, overlapping speech, and noisy or accented audio.

**Consumes.** The clean speech stem, the diarization boundaries (transcribing per-speaker turn improves accuracy), and any project glossary (names and terms to bias toward).

**Produces.** A transcript with per-word timestamps, per-word confidence, punctuation, casing, and detected disfluencies. Confidence is a first-class output, not a nicety: it drives where humans should look.

**Interactions.** Feeds linguistic analysis and translation; its timestamps feed the timing planner; its confidence feeds QC and human review routing.

**Tradeoffs.** Transcript is the cheapest place in the entire pipeline to correct an error and the highest-leverage place to do so, because a wrong word here corrupts translation, timing, and synthesis simultaneously. This argues strongly for a human checkpoint on the transcript for high-tier content (§13), and for surfacing low-confidence words prominently. Forced alignment as a refinement pass tightens word boundaries even when the recognizer's raw timestamps are loose, which materially improves downstream timing.

### 3.6 Linguistic and semantic analysis

**Purpose.** Turn the raw transcript into translation-ready, context-aware units.

**Problems solved.** Translation quality depends on translating coherent units with enough surrounding context, with consistent terminology, and with awareness of register and discourse. Segment boundaries chosen for diarization are not always the right boundaries for meaning.

**Consumes.** The transcript and the speaker map.

**Produces.** Sentence and clause segmentation aligned to the segment skeleton, named entities (names, places, brands to lock), terminology candidates, register/formality signals, and discourse markers (who is addressing whom, questions versus statements) that help translation and prosody.

**Interactions.** Feeds translation directly and feeds prosody analysis (a question's intonation, an emphasized entity).

**Tradeoffs.** Re-segmenting for meaning can conflict with the timing skeleton; the design keeps a mapping between meaning-units and timing-units so a translated sentence can later be redistributed across the original timing slots. This is fiddly but necessary, because translating word-by-word or fixed-window destroys quality.

### 3.7 Translation

**Purpose.** Render the source meaning into the target language, faithfully, in a style consistent with the speaker and the content, and (this is the part generic MT ignores) within length and rhythm constraints compatible with the timeline.

**Problems solved.** Beyond ordinary translation quality, dubbing translation must respect: timing feasibility (a translation that cannot be spoken naturally in the available time is a bad translation, regardless of its accuracy), terminology and name consistency, register and character voice, and cultural adaptation (idioms, honorifics, jokes).

**Consumes.** Meaning-units with context, the project glossary and translation memory, register signals, and crucially the **available time budget per segment** from the timing planner (this is the feedback loop).

**Produces.** Target text per segment, ranked candidates (not a single output), and per-candidate metadata: estimated spoken duration, length-compliance against the budget, glossary compliance, and a quality estimate.

**Interactions.** This is the stage most tightly coupled to timing. It does not run once and finish; it participates in the joint optimization (§3.9). It reads and writes translation memory for long-form consistency (§7).

**Tradeoffs.** Length-controlled translation trades a little literal fidelity for spoken-ability, and the system should make that trade *visible* and *tunable*. Generating multiple candidates costs more compute but is what lets the timing planner choose a translation that fits rather than forcing a fit onto a single rigid output. Context windows must be long enough for coherence across a scene (and a glossary/TM for coherence across the whole work), which is a cost worth paying because inconsistency across a long piece is one of the most noticeable failures.

### 3.8 Prosody and emotion analysis

**Purpose.** Capture *how* each line was delivered (its emotion, emphasis, tempo, and intonation) so synthesis can reproduce the delivery rather than reading the translation flatly.

**Problems solved.** Identity without delivery is a flat, robotic dub. The viewer's sense of performance lives in prosody: pitch movement, energy, timing of stress, pauses, and emotional coloring. This must transfer across languages even though the literal pitch contour cannot be copied (different words, different phoneme counts, different stress rules).

**Consumes.** The source speech stem per segment, the word-level alignment, and discourse signals (a question, an exclamation).

**Produces.** A **prosody/affect representation** per segment: a learned latent capturing emotional state and speaking style, plus interpretable descriptors (relative pitch range, energy, speaking rate, emphasis locations mapped to source words, and pause structure). The interpretable layer matters for human editing ("make this angrier," "emphasize this word") and for the timing planner.

**Interactions.** Feeds synthesis as conditioning, feeds the timing planner (pause structure and natural rate inform the duration plan), and is editable by humans.

**Tradeoffs.** The crucial design choice is to transfer *style and affect*, not raw acoustic contours. Copying the source F0 curve onto target words sounds wrong, because prosody is language-specific. Instead the system extracts a higher-level representation (what emotion, how much emphasis, where) and lets a multilingual synthesizer realize that representation according to the target language's prosodic norms. This is genuinely hard and not fully solved at the highest fidelity; cross-lingual prosody transfer is one of the frontiers this design is honest about (§9).

### 3.9 Timing and duration planning: the joint optimizer

**Purpose.** Decide the target timing for every segment so that translated, expressively-rendered speech fits the fixed timeline as naturally as possible. This is where the central constraint of §0 is actually resolved.

**Problems solved.** The core tension: faithful translations have different durations than the slots available, and forcing them in (by speeding up speech beyond natural limits) destroys quality, while leaving them loose breaks sync and pacing.

**Consumes.** Per segment: the available slot (from source timing and scene structure), the candidate translations with their estimated durations (from translation), the natural speaking rate and pause structure (from prosody analysis), and scene-level structure (where cuts are, where silence exists that can absorb slack).

**Produces.** A **timing plan** per segment: chosen translation candidate, assigned start and duration, speaking-rate factor (kept within natural bounds, roughly 0.9x to 1.15x as a default tolerance), and pause budget. Where no candidate fits, it produces either a request back to translation for a shorter candidate or a flag that this segment requires a human decision or a scene-level accommodation.

**Interactions.** This is the hub of the joint optimization. It pulls candidates from translation, pushes length constraints back to translation, and once settled, hands the final per-segment target (text, timing, prosody) to synthesis. It is the single place where the translation/prosody/timing triad is reconciled.

**The optimization itself.** Conceptually this is constrained optimization over the scene: choose, for each segment, a (translation candidate, rate factor, pause allocation) that minimizes a cost combining translation-quality loss, deviation from natural speaking rate, and pause/timing distortion, subject to segments not colliding and aligning to cuts. Several levers expand the feasible set before any quality is sacrificed:

- **Compress or redistribute pauses** rather than the speech itself (listeners tolerate pause changes far better than rate changes).
- **Let segment boundaries shift** within a turn so slack can move to where it is least audible.
- **Allow scene-level micro time-stretch of the video** (a fraction of a percent, imperceptible) to buy time across a stretch where dialogue is dense. This trades a tiny, usually invisible video manipulation for a large gain in speech naturalness, and should be a configurable knob.
- Only after these are exhausted does the planner spend the speaking-rate budget, and only after *that* does it ask translation for a more compressed (less literal) candidate.

**Tradeoffs.** This is the most important and most distinctive subsystem. The alternative (translate first, then desperately time-stretch to fit) is what makes mediocre dubs sound rushed and unnatural. The cost is that translation and timing cannot be cleanly separated stages with a clean handoff; they need a feedback loop, which complicates orchestration. That complexity is the price of quality and the design pays it deliberately.

### 3.10 Voice synthesis (cross-lingual cloning TTS)

**Purpose.** Generate target-language speech in each speaker's cloned voice, conditioned on the delivery and timing the upstream stages decided.

**Problems solved.** This is the heart: produce speech that *is* the speaker's voice, *in a language they may never have spoken*, with the *right emotion and emphasis*, hitting the *assigned duration*. Each of those four requirements is conditioning that the synthesizer must honor simultaneously.

**Consumes.** Per segment: the target text, the speaker voice profile and reference set, the prosody/affect representation, and the timing plan (target duration and rate).

**Produces.** Synthesized speech audio per segment, plus the actual realized duration and an internal confidence/quality signal.

**Interactions.** Reads from the four upstream sources above; writes per-segment renders into the content-addressed store, keyed so they are reused unless their inputs change (§11).

**Tradeoffs and the central difficulty.** The four conditioning signals are in tension. Honoring a tight duration can degrade naturalness; honoring strong emotion can drift the timbre away from the speaker; honoring the speaker identity from an accented reference can leak accent. The design's response is disentangled conditioning (separate channels for identity, content, prosody/affect, and language so the model can vary one without disturbing the others) and per-speaker model strategy (fine-tuned models for leads, zero-shot for minor speakers). Even so, the joint satisfaction of identity plus cross-lingual prosody plus exact duration is at the edge of current capability, which is why the timing planner works to *hand synthesis a duration it can hit naturally* rather than demanding the synthesizer absorb all the timing stress.

### 3.11 Speech post-processing and fine alignment

**Purpose.** Take per-segment renders to broadcast-clean, exactly-timed audio.

**Problems solved.** Even a good render lands a few percent off its target duration, may have minor artifacts, and needs to be placed precisely on the timeline. Forcing exact duration inside the synthesizer hurts quality, so a light external correction is preferable.

**Consumes.** Per-segment renders and their timing plans.

**Produces.** Final per-segment dialogue audio, time-corrected (sub-segment time-stretching using high-quality, pitch-preserving methods to absorb the last small percentage), de-clicked, de-essed where needed, and level-normalized, placed at exact start times.

**Interactions.** Hands a complete dialogue stem (all segments, placed) to the remix stage; can also hand timing residuals to lip-sync if visual dubbing is enabled.

**Tradeoffs.** A small external time-stretch (a few percent) is inaudible and far cheaper than retrying synthesis; large stretches are audible and should instead trigger a re-plan or re-synthesis. The split of responsibility (synthesizer gets close, post-processing nudges) keeps both stages in their comfort zone.

### 3.12 Background, music, and effects preservation, then remix and master

**Purpose.** Reconstruct the full soundtrack by placing the new dialogue into the original sonic world, then master to delivery standards.

**Problems solved.** The new voice must sound like it was recorded in the same space as the scene, sit at the right level against music and effects, and the overall mix must meet loudness and peak standards for the target platform.

**Consumes.** The placed dialogue stem (from post-processing) and the preserved music, effects, and ambient stems (from separation, untouched).

**Produces.** A final mixed and mastered audio track (or several, for multi-track delivery), meeting target loudness (for example a platform LUFS target) and true-peak limits, plus the original-language and any music-and-effects-only tracks as deliverable artifacts.

**Interactions.** This is the convergence point of the two halves of the pipeline (the rebuilt dialogue and the preserved background). It hands the finished audio to the mux stage.

**The often-overlooked details.** Three things separate a convincing mix from an obviously-dubbed one, and the design treats each as explicit:

- **Acoustic-environment matching.** Synthesized dialogue is typically dry (studio-clean). The original dialogue carried the room: reverb, distance, telephone or radio coloration. The remix must re-apply matching acoustic treatment so the new voice sits in the same space. Estimating that environment from the original speech stem and re-applying it is a real subsystem, not a slider.
- **Ducking and level automation.** Music and effects must duck appropriately under the new dialogue, following the new dialogue's timing, not the original's.
- **Loudness and dynamics.** Final mastering to the delivery target, preserving the dynamic character of the original rather than flattening it.

**Tradeoffs.** Re-applying environment risks doubling reverb if the dialogue stem was not fully dry; the cleaner the separation, the better this works, which links mix quality back to separation quality. Where original music-and-effects stems are supplied, the remix is dramatically better, another reason to support bring-your-own-stems.

### 3.13 Lip-sync and visual dubbing (optional, tiered)

**Purpose.** Address the visual mismatch between the original mouth movements and the new speech.

**Problems solved.** When the camera shows a speaking face, the new audio's phonemes do not match the visible mouth. This ranges from unnoticeable (off-screen, wide shots, fast cuts) to glaring (sustained close-ups).

The design offers three tiers, as an explicit configuration choice, because they differ enormously in cost, risk, and quality:

- **Tier 1, audio-only with timing respect.** Do nothing to the video; rely on the timing planner having aligned speech onsets and overall duration to the original as closely as possible. This is the default and is acceptable for a large fraction of content. Honest about its limit: close-ups will show mismatch.
- **Tier 2, timing-to-lip refinement.** Bias the timing planner to align the *new* speech's mouth-relevant events (syllable onsets, especially bilabial closures) to the *original* lip motion where a face is on screen, trading some prosodic freedom for better apparent sync without altering pixels.
- **Tier 3, visual dubbing.** Regenerate or modify the speaker's mouth region to match the new phonemes (face reenactment). This is the highest fidelity and by far the most fraught: it is computationally heavy, it can land in the uncanny valley, and it raises serious consent and provenance questions (§17). It must be consent-gated and watermarked.

**Consumes.** Original frames, the speaker's face track, and the final dialogue audio (Tier 3 also needs phoneme timing).

**Produces.** Either nothing (Tier 1), timing constraints fed back to planning (Tier 2), or modified video frames (Tier 3).

**Interactions.** Tier 2 couples back into the timing planner. Tier 3 sits between speech post-processing and the mux, producing new frames.

**Tradeoffs.** This is the clearest place where the system must refuse to pretend. Tier 3 visual dubbing is genuinely hard to do without artifacts on arbitrary content, and doing it on a real person's face without consent is not an engineering decision, it is an ethical and legal one. The architecture makes it opt-in, gated, and watermarked, and defaults to honest audio-only dubbing.

### 3.14 Video reconstruction and mux

**Purpose.** Recombine everything into deliverable files.

**Consumes.** Original (or visually-dubbed) frames, the final mastered audio track(s), and generated subtitle/caption artifacts.

**Produces.** Final deliverables per target language: muxed video with the dubbed track (and optionally the original track and a music-and-effects track), plus subtitle/caption files generated from the target text as a natural by-product.

**Interactions.** Terminal stage of the render path; reads the finished audio and (optionally) frames.

**Tradeoffs.** Generating subtitles for free from the target text is a nice efficiency, but dubbing subtitles and reading subtitles have different conventions (timing, condensation), so the system should generate dubbing-aligned captions and offer a separately-optimized reading-subtitle track if requested.

---

## 4. Speaker identity preservation (consolidated)

Pulling the identity thread together, because it spans several stages and is one of the two or three things that most determine whether a dub feels real.

Identity preservation is a chain, and the chain is only as strong as its weakest link:

1. **Correct attribution** (diarization): the right voice is even being used. Multimodal (video-assisted) diarization is the main lever.
2. **Clean, representative reference** (profiling and reference selection): the clone has good material to imitate. The reference selector and its explicit no-good-reference failure state guard this.
3. **Disentangled voice representation**: identity is captured without entangling source language or accent, so cross-lingual cloning does not leak an accent.
4. **Identity-preserving synthesis under stress**: when the synthesizer is pushed on emotion or duration, timbre must not drift. Disentangled conditioning and fine-tuning leads protect this.
5. **Consistency over time** (speaker registry, §7): the same character sounds the same in minute 3 and minute 93, and in episode 1 and episode 40.

The honest difficulty is link 3 plus 4 together: preserving identity *and* cross-lingual delivery *and* hitting timing, simultaneously, from imperfect references. The design's strategy is to (a) invest heavily in reference quality so the clone starts from good material, (b) use disentangled representations so the factors do not fight, (c) fine-tune principal speakers where data allows, and (d) offload timing stress to the planner so the synthesizer is asked for durations it can hit without distorting the voice.

---

## 5. Cross-lingual transfer, the underlying mechanism

The reason identity and emotion can transfer across languages at all is representational disentanglement. The system models speech as a small set of factors:

- **Speaker identity** (who): timbre and vocal-tract characteristics.
- **Linguistic content** (what): the phonetic/text content, language-specific.
- **Prosody and affect** (how): emotion, emphasis, tempo, language-agnostic at the affect level.
- **Language and accent**: a factor the system wants to *set to the target*, not inherit from the source.

If these are truly separable, dubbing becomes: keep identity, keep affect, swap content and language. Reality is messier (the factors are not perfectly independent, and prosody is partly language-bound), but architecting around this factorization is what makes the cross-lingual problem approachable rather than magical. It also gives clean handles for human editing: change the translation (content) without touching the voice (identity) or the performance (affect).

---

## 6. Emotion, prosody, intonation, and speaking-style preservation (consolidated)

These four named requirements are facets of one thing: transferring the performance. The design treats them with one representation that has both a learned and an interpretable layer.

- **Emotion** is the affective state (angry, tender, excited). Captured in the affect latent; editable via interpretable descriptors.
- **Prosody** is the broader umbrella: the rhythm, stress, and melody of speech. Represented as relative pitch range, energy, rate, and pause structure, mapped to words.
- **Intonation** is specifically the pitch melody (the rise of a question, the fall of finality). Handled as relative pitch movement realized per the target language's rules, never as a copied absolute contour.
- **Speaking style** is the speaker's habitual manner (clipped, drawling, formal, breathy). Partly identity (it lives near timbre and is captured by the voice profile) and partly affect.

The governing principle, repeated because it is the crux: **transfer the high-level intent, realize it in the target language's own prosodic system.** Copying source acoustics onto target words is the classic mistake. The interpretable layer exists so that when the automatic transfer is imperfect (and on subtle performances it will be), a human can nudge "more emphasis here, softer there" without re-recording anything.

This is, candidly, the area where automatic systems are furthest from human dubbing directors. A skilled voice director coaxes a performance that matches the original's intent in a way no current model reliably matches. The design's posture is to get close automatically and make the gap cheap to close by hand, rather than to claim the gap does not exist.

---

## 7. Long-form and franchise-level consistency

A two-hour film, a 200-episode series, a multi-year YouTube channel: consistency across all of it is its own engineering problem, separate from per-segment quality, and one of the most noticeable when it fails.

The mechanisms:

- **Speaker registry scoped above the project.** Voices live in a registry that can be scoped to a single work *or* to a franchise/channel. Character "Alice" is linked to a stable voice profile that is reused across every episode, so she sounds identical throughout. New episodes link to existing registry entries rather than re-deriving voices from scratch. This is the "voice bible."
- **Translation memory and glossary scoped above the project.** Names, invented terms, catchphrases, and recurring lines are stored and enforced consistently across the whole work and franchise. "The Iron Bank" is translated the same way every time it appears, across every episode.
- **Style and register locking.** Character voice in translation (formal versus casual, characteristic verbal tics) is captured as per-character translation guidance and applied consistently.
- **Cross-episode speaker linking.** When a new episode is processed, its diarized speakers are matched against the franchise registry by voice embedding, so recurring characters are recognized automatically (and flagged for human confirmation when ambiguous).

The architectural enabler is that the speaker registry, translation memory, and glossary are not per-job artifacts; they are persistent, shared domain objects that jobs read from and contribute back to. This is what turns a collection of independently-dubbed episodes into a coherent dubbed *series*.

---

## 8. Cumulative error: how quality degrades and how to minimize it

Every stage is lossy, and errors compound multiplicatively down the chain. An ASR error becomes a translation error becomes a timing error becomes a synthesis error, and the viewer hears all of it at once. Naming the principal degradation sources and the design's response to each:

- **Separation bleed** corrupts both recognition and reference quality. Response: use cleanest regions for the highest-stakes purposes, support bring-your-own-stems, surface low-confidence regions.
- **ASR errors** corrupt everything textual. Response: confidence as a first-class output, forced-alignment refinement, and a human checkpoint at the transcript (the cheapest, highest-leverage place to fix anything).
- **Translation/timing infeasibility** forces unnatural speech. Response: the joint optimizer with candidates and feedback, so quality is traded knowingly and minimally rather than forced.
- **Synthesis drift** under emotion or duration stress. Response: disentangled conditioning, offloading timing stress to the planner, fine-tuning leads.
- **Mix mismatch** breaks the illusion at the end. Response: explicit acoustic-environment matching and timing-aware ducking.

Four cross-cutting principles keep cumulative error down:

1. **Carry the source forward, not just its text.** Later stages keep access to the original audio and rich features, not a lossy text summary, so they can reference ground truth (for example, prosody analysis works from the actual audio, not from a guess off the transcript).
2. **Couple tightly-coupled stages instead of pipelining them.** The translation/timing loop is the prime example. Where information loss across a clean handoff would be severe, fuse the stages.
3. **Propagate confidence and route uncertainty to humans.** The system always knows where it is unsure (low ASR confidence, infeasible timing, no good reference, ambiguous diarization) and surfaces exactly those spots, so human effort lands where it matters.
4. **Checkpoint at cheap, high-leverage stages.** A human minute spent fixing a transcript or a translation saves an hour of confusing downstream symptoms. The workflow (§13) is built around these checkpoints.

---

## 9. The hardest technical challenges, and how to mitigate them

A consolidated, honest list. None of these is fully solved; each has a mitigation that makes it tractable rather than perfect.

1. **Isochrony versus naturalness.** The defining constraint. *Mitigation:* the joint timing optimizer with multi-candidate translation, pause-first and boundary-shift levers, optional imperceptible scene-level video time-stretch, and a strict speaking-rate budget spent only as a last resort. Expose a faithfulness-versus-timing knob.

2. **Cross-lingual identity preservation without accent leakage.** *Mitigation:* disentangled speaker representations, aggressive reference-quality curation with an explicit failure state, and fine-tuning for principal speakers.

3. **Cross-lingual prosody and emotion transfer.** The furthest from human parity. *Mitigation:* transfer high-level affect rather than raw contours, realize it per target-language norms, and provide an interpretable editing layer so humans can close the residual gap cheaply. Be honest that subtle performances will need human direction.

4. **Source separation in dense mixes.** *Mitigation:* best-available demixing, cleanest-region routing for high-stakes uses, bring-your-own-stems as a first-class path, and surfacing the hard regions rather than hiding them.

5. **Diarization errors (wrong voice for a line).** Very audible. *Mitigation:* multimodal diarization fusing video active-speaker detection with audio and ASR cues; flag and route ambiguous turns.

6. **Long-form and franchise consistency.** *Mitigation:* persistent, shareable speaker registry, translation memory, and glossary; cross-episode speaker linking by embedding.

7. **Cumulative error.** *Mitigation:* carry source forward, couple tight stages, propagate confidence, checkpoint cheap high-leverage stages (§8).

8. **Lip-sync.** *Mitigation:* tiered and optional, defaulting to honest audio-only; visual dubbing consent-gated and watermarked.

9. **Acoustic-environment matching.** Often the final tell. *Mitigation:* estimate and re-apply the original speech's acoustic environment to the synthesized dialogue in the remix.

10. **Evaluation itself.** There is no clean reference for "the correct dub," so objective metrics are partial proxies. *Mitigation:* a portfolio of objective metrics plus disciplined subjective evaluation, with no single metric trusted alone (§20).

---

## 10. Scalability and the execution model

The processing stages have wildly different resource profiles: separation, ASR, synthesis, and visual dubbing are GPU-heavy; translation is GPU or external API; orchestration, IO, and mixing are CPU-bound. The execution model is built around that heterogeneity.

- **Stage-typed worker pools.** Each stage type has its own autoscaling pool sized to its hardware. GPU pools scale independently of CPU pools. Expensive stages (visual dubbing) get their own pool with its own limits.
- **Queue-based orchestration with backpressure.** Stages communicate through durable queues. The DAG engine enqueues work as upstream artifacts become available. Backpressure prevents a fast stage from overwhelming a slow expensive one.
- **Embarrassing parallelism after planning.** This is the key scalability insight: once the timing plan is fixed, *every segment's synthesis and post-processing is independent.* A feature film is thousands of independent segment renders. They fan out across the synthesis pool with near-linear speedup. The serial part (separation, diarization, ASR, the translation/timing loop) is a small fraction of total compute; the bulk (synthesis) parallelizes freely.
- **Chunked long-form processing.** Long content is split into scenes (at natural cut/silence boundaries), processed in parallel, and stitched. Cross-scene consistency comes from the shared registry/TM/glossary, not from processing serially.
- **Idempotency everywhere.** Every stage is keyed by the content hash of its inputs (§11), so retries and duplicate deliveries are safe, and horizontal scaling needs no coordination beyond the queue.

---

## 11. Incremental processing and resumability

This is where the content-addressed Timeline pays off, and where a reader from a build-systems background will recognize the pattern: this is essentially an incremental build graph for media.

- **Content-addressed artifacts.** Every artifact (a stem, a transcript, a segment render) is stored under a hash of its inputs plus the configuration and model versions that produced it. Identical inputs yield a cache hit; nothing is recomputed.
- **Segment-level invalidation.** Because the Timeline is segment-granular and each segment's render key depends only on that segment's synthesis-relevant fields, editing one line's translation invalidates exactly that line's render and its mix contribution, and nothing else. Re-rendering after a one-word fix touches one segment, not the movie.
- **Resumable jobs.** A job that fails or is paused resumes from the last completed artifact boundary. Because stages are idempotent and artifacts are cached, "resume" is just "re-run the DAG; completed nodes are cache hits."
- **Cheap experimentation.** Branching the Timeline to try an alternate translation of a scene re-renders only that branch's changed segments, making A/B exploration inexpensive.

The mental model is exactly that of a correct incremental build system: a dependency graph, content-addressed outputs, and minimal recomputation on change. Applying it to a dubbing pipeline is what makes human-in-the-loop iteration fast enough to be usable, because human edits trigger surgical re-renders rather than full reprocessing.

---

## 12. Failure recovery

- **Idempotent, content-keyed stages** make retries always safe; transient failures (a GPU OOM, a node eviction) are simply retried, often landing on cache hits for already-completed sub-work.
- **Per-segment isolation.** A single bad segment (synthesis fails on one line) is retried or routed to review without failing the whole job. Partial progress is preserved by construction because artifacts are cached as they complete.
- **Graceful degradation with flags.** If an optional stage fails (visual dubbing), the system falls back to the next tier (audio-only) and *flags the degradation* rather than failing or silently shipping a worse result. The user is told what was degraded and why.
- **Dead-letter and human routing.** Work that fails repeatedly, or that the system is not confident about, goes to a review queue rather than to a silent default. The default behavior under uncertainty is "ask a human," not "guess."
- **Poison-input handling.** Inputs that crash a stage are quarantined with diagnostics rather than retried forever.

---

## 13. Human-in-the-loop editing workflows

Fully automatic dubbing is good enough for some tiers and not for others. A world-class platform treats humans as a configurable, high-leverage part of the pipeline, not an admission of failure. The design places editing at the stages where human effort has the most leverage per minute spent.

The leverage-ordered checkpoints:

1. **Transcript review** (highest leverage). Fixing a misrecognized word here prevents cascading errors in translation, timing, and synthesis. Low-confidence words are highlighted for fast scanning.
2. **Speaker assignment review.** Confirm or correct diarization, especially ambiguous turns and no-good-reference speakers. Cheap to do, very audible if wrong.
3. **Translation review.** Adjust meaning, register, character voice, and cultural adaptation. The editor sees source and target side by side, with the timing budget and length-compliance shown so the editor understands the constraint they are working within.
4. **Performance review.** Audition synthesized lines; nudge emotion and emphasis through the interpretable prosody layer; re-synthesize a single line on demand (a fast, segment-scoped re-render thanks to §11).
5. **Mix review.** Adjust dialogue level, ducking, and environment matching for problem moments.

Because every edit is a mutation of the canonical Timeline and triggers only segment-scoped re-rendering, the loop is tight: edit a line, hear the result in seconds. The same Timeline-mutation operations back the API, CLI, and UI, so a power user batch-editing via CLI and an editor working in the UI are doing the same thing through different surfaces.

---

## 14. Storage architecture

Storage is differentiated by access pattern:

- **Object store for large media** (source files, stems, segment and final renders, video frames), content-addressed, with lifecycle policies (hot for active projects, cold/archive for finished ones). This holds the bulk of the bytes and is the backing store for the artifact cache.
- **Metadata database for the Timeline and project state.** The Timeline is structured, queryable, versioned data, not a blob. It needs transactional updates (a human edit must be atomic), version history, and efficient segment-level access. A document or relational store with strong consistency fits.
- **Vector store for embeddings.** Speaker embeddings (for diarization, registry linking, and cross-episode matching) and translation-memory embeddings (for fuzzy reuse) live in a similarity-searchable index.
- **Append-only event log.** Every stage execution and every human edit is recorded for auditability, debugging, reproducibility, and provenance (which matters legally and ethically, §17). This also makes "explain how this output was produced" answerable.

The separation matters because conflating them (for example, putting the Timeline in the object store as a blob) destroys the very properties (segment-level invalidation, transactional editing, queryability) that the rest of the design relies on.

---

## 15. Metadata architecture

Metadata is not decoration here; it is the system's nervous system. Three tiers:

- **Project and workspace metadata.** A *workspace* (tenant) contains *projects*; a project is one work (or a series sharing a registry). Project metadata holds configuration (target languages, quality tier, tier choices like fine-tune-leads or visual-dubbing-on), the probe report, and processing history.
- **The Timeline** (§2), the operational metadata: segments and everything attached to them. This is what stages read and write and what humans edit.
- **Shared domain metadata** (§7): the speaker registry, translation memory, and glossary, scoped to project or franchise, persistent across jobs.

Two principles: metadata is **versioned and immutable-by-append** (you can always reconstruct any past state and explain any output), and metadata is **the contract between stages and surfaces** (stages agree on the Timeline schema, not on each other's internals, so a stage can be reimplemented without touching its neighbors).

---

## 16. API architecture

The API is the core; CLI and UI are clients of it. Its shape follows the domain model.

- **Project and workspace resources.** Create projects, configure them, manage the shared registry/TM/glossary.
- **The Timeline as the central editable resource.** Read the Timeline; patch segments (translation, speaker, prosody params, timing, locks); branch and version it. This is the most-used surface and it is the same operation set the UI's editor uses.
- **Jobs and orchestration.** Submit processing (full pipeline or a single stage or a re-render of a segment), query status, subscribe to progress. Jobs are asynchronous, with streaming progress and webhook callbacks, because a feature film does not dub in one request cycle.
- **Artifact access.** Retrieve any artifact (stems, renders, deliverables) by reference, via signed URLs for the large media.
- **Granularity.** The API exposes the same granularity the engine has: you can re-run one stage, re-render one segment, or run everything. This is what lets external tooling and CI build on the platform rather than treating it as a black box.

The design principle is that anything the UI can do, the API can do, because the UI is built on the API. There is no privileged path.

---

## 17. CLI architecture

The CLI is a thin, scriptable client over the API, aimed at power users, automation, and CI.

- **Maps to the domain:** create a project, configure languages and tiers, run the pipeline or a single stage, edit the Timeline (including batch edits and glossary imports), audition and re-render segments, and pull deliverables.
- **Scriptable and resumable.** Commands are composable and idempotent; a CI pipeline can dub a backlog of videos, and a re-run skips completed work via the content-addressed cache. Long jobs run asynchronously with status polling or streaming.
- **Honest output.** The CLI surfaces confidence, flags, and degradations (what got routed to review, what was degraded and why), so automated callers can decide whether human review is needed before publishing.

Because it goes through the same API and the same Timeline mutations, the CLI is never out of step with the UI; they are two faces of one core.

---

## 18. Future UI architecture

The UI is, fundamentally, a timeline editor for dubbing: a non-linear-editor / digital-audio-workstation experience specialized for this domain. Its anatomy:

- **A timeline view** with the waveform, scene/cut markers, and per-speaker tracks, scrubbing the source and the dub in sync.
- **A segment inspector** showing source text and target text side by side, with the timing budget and length-compliance visible, the prosody/affect controls (the interpretable layer) exposed as direct manipulations, and an audition/re-synthesize control that triggers a segment-scoped re-render.
- **A speaker/cast panel** for managing the registry, confirming diarization, and assigning or fine-tuning voices.
- **A glossary and translation-memory panel** for franchise consistency.
- **A review and flags panel** that takes the editor directly to the spots the system flagged (low confidence, infeasible timing, no good reference), so human attention follows the system's uncertainty.

It is built entirely on the API, editing the same Timeline. Real-time-feeling iteration is possible precisely because of segment-level incremental rendering (§11): the editor changes one line and hears it in seconds, which is the difference between a usable tool and a frustrating one. The UI adds no dubbing logic; it is an ergonomic front end to the same operations the API and CLI expose.

---

## 19. Testing strategy

Testing a probabilistic, multi-stage media system needs more than unit tests, though it needs those too.

- **Unit and contract tests** at stage boundaries: each stage is tested against the Timeline schema it consumes and produces, so stages can be reimplemented independently. Deterministic logic (timing math, orchestration, caching/invalidation) is unit-tested hard, because a caching bug that returns stale renders is insidious.
- **Golden datasets** spanning the axes that matter: languages and language pairs, genres (drama, documentary, comedy, action), speaker counts (one to dozens), audio conditions (clean studio to noisy field), and content lengths (short clip to feature). Regression runs compare new outputs against established baselines on objective metrics, catching quality regressions before release.
- **Property and invariant tests.** For example: segment-level invalidation must re-render exactly the changed segments and no others; a re-run with no changes must be all cache hits; total dubbed duration of a scene must match the source within tolerance.
- **Fault-injection tests.** Kill workers mid-job, corrupt an input, fail an optional stage, and assert correct recovery, graceful degradation, and flagging.
- **Human-evaluation harness** (see §20), run on a fixed panel and fixed sets so subjective scores are comparable across releases.

The hardest part is that "correct" output is not unique, so much of testing is regression (did it get worse?) and invariant-checking (did it violate a hard constraint?) rather than exact-match assertion.

---

## 20. Evaluation methodology, objective and subjective

Because there is no single ground-truth dub, evaluation is a portfolio, and no metric is trusted alone.

**Objective metrics** (cheap, automatable, partial proxies):

- *Recognition:* word/character error rate on the source transcript (and round-trip checks).
- *Translation:* reference-based and reference-free quality estimates, plus dubbing-specific metrics that generic MT ignores: length-compliance against the timing budget and terminology/glossary compliance.
- *Voice similarity:* cosine similarity between the synthesized voice's embedding and the speaker's reference embedding (identity preservation, quantified).
- *Speech naturalness:* learned mean-opinion-score predictors and artifact detectors (clicks, glitches, unnatural prosody).
- *Prosody and timing:* per-segment duration error, speaking-rate distribution (is the dub being forced too fast?), pause alignment, and pitch-movement correlation where meaningful.
- *Sync:* audio-visual synchronization confidence from a sync-estimation model, and onset alignment to the original.
- *Mix:* loudness (LUFS), true-peak compliance, and stem-balance checks.

**Subjective metrics** (expensive, but the ground truth that the objective metrics approximate):

- Mean-opinion-score panels for *naturalness*, *intelligibility*, and *speaker similarity*.
- Targeted ratings for *emotion match* and *identity preservation*, the things automatic metrics capture worst.
- *A/B comparisons* against human dubs and against prior system versions.
- *Overall viewing-experience / immersion* ratings on full clips, not isolated lines, because dubbing quality is partly a whole-scene property.
- *Native-speaker evaluation* of translation quality and cultural appropriateness, which no automatic translation metric fully captures.

**Methodology discipline:** fixed evaluation sets and fixed rater pools for comparability; enough raters and reported confidence intervals for statistical validity; per-language calibration, since both the objective metrics and rater expectations differ across languages. Objective metrics gate every build (fast, automatic); subjective evaluation gates major releases (slow, authoritative). The two are kept correlated by periodically checking that objective movements predict subjective ones, and recalibrating the objective suite when they diverge.

---

## 21. Production deployment considerations

- **GPU capacity is the dominant cost and the main scaling constraint.** The deployment must pool GPUs across tenants, schedule by stage type, and exploit the fact that the heavy stage (synthesis) is embarrassingly parallel, so it scales horizontally and can absorb spot/preemptible capacity (retries are safe by idempotency).
- **Asynchronous, long-running jobs** are the norm; the platform is built around durable queues, progress streaming, and webhooks, not synchronous request/response. A dubbing job is minutes to hours, not milliseconds.
- **Multi-tenancy and isolation:** per-workspace data isolation, fair scheduling so one large job does not starve others, and per-tenant quotas.
- **Provenance, consent, and disclosure are deployment requirements, not nice-to-haves.** Voice cloning of real people and visual dubbing of real faces carry consent and likeness obligations. The platform should: gate cloning and visual dubbing behind consent records, watermark synthetic audio (and any regenerated video) for downstream detectability, and keep the append-only provenance log so any output's origin is auditable. Honest synthetic-media disclosure is part of the product, not an afterthought.
- **Observability:** per-stage latency, failure, and quality-metric dashboards; alerting on quality regressions caught by the objective suite; the event log as the backbone of debugging and audit.
- **Cost controls:** the content-addressed cache is also a cost control (never recompute), and the quality tiers are cost tiers (zero-shot versus fine-tune, audio-only versus visual dubbing) the operator and user can choose deliberately.

---

## 22. Short-form versus long-form, and one versus many speakers

Two scaling axes the architecture must span without separate codebases.

**Short-form to long-form.** The same DAG handles both; the differences are degree, handled by chunking and shared state. Short content runs as a single chunk with low latency. Long content is split at scene boundaries, processed in parallel (§10), and stitched, with cross-chunk coherence supplied by the shared speaker registry, translation memory, and glossary (§7) rather than by serial processing. The serial, coupled core (separation through the translation/timing loop) is a small fraction of compute; the parallel bulk (synthesis) scales out, so a feature film is mostly a throughput problem, not a latency problem. Incremental rendering (§11) keeps long-form *editing* fast: fixing one line in a two-hour film re-renders one line.

**One speaker to dozens.** Diarization and the speaker registry generalize across speaker count by construction. The per-speaker model strategy is the lever that keeps quality and cost sane as speaker count grows: fine-tune the few principal speakers (who carry most of the runtime and most of the audience's attention), zero-shot the long tail of minor speakers (who appear briefly and where zero-shot quality suffices). Multimodal diarization is what keeps attribution reliable as the cast grows and overlaps multiply. The honest limit is dense overlapping crowd dialogue, which is hard for separation and diarization alike; the system flags those regions rather than pretending to resolve them perfectly.

**Consistency across a whole work or franchise** (movie, podcast, episode, documentary, channel) is the unifying requirement across both axes, and it is solved by the shared, persistent domain objects of §7: one voice per character everywhere, one translation per term everywhere, recognized automatically across episodes by embedding match and confirmed by humans when ambiguous. That is what turns per-segment quality into a coherent dubbed body of work.

---

## 23. Where this design is honest about its limits

A closing summary of the frontier, because a design document that claims everything is solved is not trustworthy.

- **Cross-lingual prosody and emotion transfer** is the area furthest from human-director quality. The design gets close automatically and makes the residual gap cheap to close by hand, but it does not claim parity with a skilled voice director on subtle performances.
- **Visual dubbing (Tier 3)** is genuinely hard to do artifact-free on arbitrary content, and carries consent obligations that are not engineering questions. It is opt-in, gated, watermarked, and off by default.
- **Source separation in dense mixes** has no perfect solution; the design routes around it (cleanest-region use, bring-your-own-stems) and surfaces the hard regions rather than hiding them.
- **Evaluation** has no clean ground truth; the design uses a portfolio of proxies and disciplined human evaluation, trusting no single number.

What the design *does* claim is that the architecture is right: a Timeline-centered, content-addressed, joint-optimizing, multimodal, human-in-the-loop system, honest about its uncertainty and built to make human effort land exactly where the machine is unsure. That backbone is what makes a state-of-the-art result achievable today, and what makes the system improve gracefully as each individual stage's models improve, because the stages are decoupled behind a stable data contract and can be upgraded one at a time.

---

### Appendix: the internal artifacts, end to end

A consolidated list of the artifacts the pipeline generates, which double as the resumability checkpoints and the human-review surfaces:

- Normalized video essence; normalized audio essence; probe report; captured source subtitles.
- Separated stems (dialogue, music, effects, ambient) and the separation quality map.
- Diarized speaker-segmented timeline; the project speaker registry.
- Per-speaker voice profiles, curated reference sets, and reference quality scores.
- Source transcript with word-level timestamps and confidence; disfluency annotations.
- Linguistic analysis: meaning-unit segmentation, named entities, terminology, register signals.
- Translation candidates per segment with length-compliance and quality estimates; translation-memory and glossary state.
- Prosody/affect representations per segment (learned latent plus interpretable descriptors).
- Timing plans per segment (chosen candidate, start, duration, rate factor, pause budget).
- Per-segment synthesized renders; realized durations; synthesis confidence.
- Post-processed, exactly-placed dialogue stem.
- Acoustic-environment estimate; the final mixed and mastered track(s); music-and-effects-only and original-language tracks.
- Optional visually-dubbed frames.
- Final muxed deliverables per language; dubbing-aligned captions and optional reading-subtitle tracks.
- The versioned Timeline (the source of truth tying all of the above together) and the append-only processing/provenance log.
