Claude finished the response

It would be nice to include a line to tell the user what age he could actually retire at, given the calculations. Say he put in age 50, but actually he could feasibly retire at 45

That's a genuinely valuable feature, but worth being careful about how it's framed — "here's the earliest age you could retire" is a much bigger claim than anything else this tool currently makes, and it needs the same rigour (or more) as everything else that's been built this session.

i dont understand the question and options

Fair — let me explain both without the jargon.

Question 1: what does the tool hold steady while it searches for the earliest retirement age?

To find "you could retire at 45," the tool has to run the simulation over and over, trying different ages, until it finds the earliest one that still passes. But everything else about the plan (spending, mortgage, super contributions) has to stay the same while it does that search — otherwise it's not really answering "how early could I retire," it's changing multiple things at once and the answer becomes murky.

    Option 1 (just search retirement age): Keep spending, mortgage, everything else exactly as you entered it. Only change the retirement age, trying earlier and earlier ages until it finds the earliest one that still succeeds. Simple, one clear answer: "you could retire at 45 instead of 50, if nothing else changes."
    Option 2 (also let it adjust spending): A more advanced version — instead of just moving retirement age, it could also ask "or, if you want to retire at 50 like you said, how much could you spend?" i.e. solve for spending level instead of age. This is a second, related but different question — more useful, but more to build.
    Option 3 (let the user pick which question they're asking): Build it so the user can choose which one they want answered — "tell me my earliest retirement age" or "tell me my max spending at my chosen age" — rather than the tool only ever answering one of those.

Given what you originally asked for ("what age could he actually retire") — that's clearly Option 1. I included 2 and 3 in case you wanted more than what you described, but I'd guess you just want Option 1.

Question 2: how does the user trigger this?

Finding the earliest feasible age isn't one simulation — it's several, because the tool needs to try age 50, then 48, then 46, then 45, etc. (or some smarter search than just guessing one year at a time) until it finds the boundary. Each of those "tries" is a full 10,000–50,000 trial run, which takes real time.

    Auto-run after every simulation: every time you run a simulation, it also automatically does this whole extra search in the background and shows you the answer. Convenient, but means every simulation run takes several times longer, even if you didn't ask for this specific answer.
    Opt-in (a menu option you choose): the normal simulation runs fast as it does now, and if you specifically want "what's my earliest retirement age," you select that from a menu afterward and wait for that specific, slower search to run.

Given this ties into the menu/navigation system (Work Item 8) already being planned, Option 2 (opt-in) fits naturally as one more item on that menu, rather than slowing down every single simulation run by default.

Which of these matches what you had in mind?

Option 1, then option 2
Add Work Item 9 for the earliest feasible retirement age search, and update deliverable counts and sequencing

Wait — that replace target doesn't exist anymore after the last edit inserted Work Item 8 before the sequencing section. Let me check the file structure.
Check the current section structure of the file

Good — it landed correctly, but there's a duplicate heading at line 137 from an earlier partial insert. Let me check and clean it up, and confirm the sequencing section is intact.
Inspect the duplicate section and surrounding context to clean it up

There's a duplicate Work Item 9 — an earlier draft (lines 137-153) is a better, more thorough version than my last insert (155-170). I'll keep the first, better one and remove the duplicate.
Remove the duplicate, less-detailed Work Item 9 section, keeping the earlier more thorough version
+1-20
Done

Now clean up the sequencing section formatting (it lost its header in the earlier edit) and verify the file end-to-end.
Viewed a file, edited a file

Now let's do a final full read-through to confirm the document is clean and consistent.
Verify final section structure is clean with no duplicates

Clean — nine work items, each properly delineated, no duplicates.
Ran a command, read a file

Added as Work Item 9, and fixed a duplication that snuck in while I was editing (two versions of the same item ended up in the file — kept the more thorough one).

A few things worth knowing about how it's scoped:

    It's a single-variable search — only retirement age moves, everything else in the client's plan is held exactly as entered. That matches what you asked for, not a broader optimiser.
    I flagged the PT income interaction explicitly — since PT income's start age defaults to retirement age (from the bug fixed earlier this session), the search needs to recalculate that window for every candidate age it tries, or it'll produce a misleading answer.
    Binary search is proposed over a brute-force year-by-year scan, but only if a monotonicity assumption holds (earlier retirement is never easier to fund) — I've made the agent confirm that assumption actually holds for this tool's mechanics before relying on it, rather than assuming it.
    Opt-in via the Work Item 8 menu, not automatic — matches your second answer, and avoids slowing down every regular simulation run.
    Multi-earner handling is flagged as an open decision, not assumed — whether the search moves both earners' retirement ages together or independently is a real scope question for the firm, not something the agent should guess at.

Results enhancement task
Document · MD 

I would like a table in the offset of the amortisation of the mortgage by year from starting year to a variable set by the user which is loan must be repaid by age x. If offset accounts have been selected, it will show clearly the year when the offset balances >= to the outstanding mortgage amount (essentially, mortgage neutral) and from then on show the sweep from these accounts into equity accounts
Add Work Item 10 for the mortgage amortisation table with offset-neutral tracking and sweep display

Now update the sequencing list and final deliverable count to include Work Item 10.
Update sequencing recommendation and deliverable count to include Work Item 10

Also update the objective line to reflect this addition.
Edited a file, ran a command

Structure's clean — ten items, no duplicates.
Presented file

Added as Work Item 10. A few things worth knowing:

    It reuses the existing overflow-sweep mechanism rather than recalculating a parallel version of it — I flagged this specifically because the earlier success-probability/worst-case bug happened when two parts of the codebase drifted apart on the same underlying number. The table needs to read whatever the simulation actually used, not approximate it separately.
    "Mortgage neutral" needs a precise definition before building — offset ≥ outstanding principal, checked at what resolution (annual vs the tool's actual monthly stepping)? I've asked the agent to state and justify this rather than pick one silently.
    Interaction with offset reserve mode is called out explicitly — if a client's using interest_cancelling mode, the offset may never be free to sweep into investments at all before the mortgage clears, since that mode locks the full principal as a floor. The table needs to reflect that constraint, not show a sweep that couldn't actually happen under the client's chosen strategy.
    Grouped with Work Items 1 and 2 in the sequencing plan, since all three fundamentally need the same thing — per-year state retained across the simulation horizon instead of discarded after each year. Worth building that retention mechanism once rather than three times.

Results enhancement task
Document · MD 


Claude is AI and can make mistakes. Please double-check responses.
Results enhancement task · MD
Project Task — Results Output Enhancement Phase
Objective

The current results output reports endpoint statistics only (success probability, horizon-balance percentiles, one worst-case trial). This phase adds trajectory visibility, tax transparency, sequencing risk isolation, scenario comparison, an earliest-feasible-retirement-age search, and a year-by-year mortgage amortisation view — turning the output from a single pass/fail verdict into something an adviser can actually walk a client through. It also adds a navigable results flow so this expanded detail doesn't have to be dumped onto one dense screen — the summary stays the default view, with a menu to drill into any additional report and return without re-running the simulation.

This is a planning task first. Do not begin implementation. For each work item below, produce a scoped implementation plan (what data needs to be captured during simulation, what changes to _SimulationState/SimulationResults are needed, what the UI/display change looks like, and what it would take to test). Flag dependencies between items, and flag anything that requires a decision from the firm before it can be scoped further. Implementation begins only after the plan is reviewed and approved.

Scope boundary, all items: the age range for every item below is the bridge period specifically — retirement age through to super/drawdown access age (the earner-specific range already used elsewhere in this tool, not a fixed 50–60; must correctly use each earner's own retirement_age and the household's super access age for households where these differ per earner).
Work Item 1 — Age-by-age trajectory (bridge period only)

Requirement: Replace or supplement the current single-horizon-point statistics with a trajectory view showing bridge asset balance at each age from retirement to super access, not just the endpoint.

Scope for the plan to address:

    What percentiles to track at each age (recommend at minimum median, P5, P95 — confirm with firm if more granularity is wanted, e.g. P10/P25/P75/P90 as well, matching the existing horizon table)
    Data capture: confirm whether per-age balances are already recorded per trial during simulation, or whether this requires a new array/list captured inside the year loop that isn't currently being retained (check memory/performance impact at 10,000–50,000 trials × up to 10+ years × multiple percentile tracking — this is a genuinely larger data volume than the current single-horizon-value approach)
    Display: table (age rows × percentile columns) vs a rendered chart — recommend a plan for both, since a chart is more legible for a client-facing conversation and a table is more precise for an adviser's own review. If a chart is proposed, confirm what's available in the tool's existing output rendering (this is a CLI tool — clarify whether chart rendering is in scope at all, or whether this should be an exported table/CSV a planner could chart externally)
    Confirm this correctly reflects each household's own bridge window when earners have staggered retirement ages (per the earlier staggered-retirement finding) — the age axis should probably be indexed by "years into the bridge" or by calendar age with a marked start/end per earner, not assume a uniform 50–60 window

Dependency: None — can be scoped independently of other items, though the underlying per-age data capture (if new) would also feed Work Item 3 (sequencing risk), so scope these together if the data model changes overlap.
Work Item 2 — Failure/near-miss depth analysis

Requirement: Supplement the single success percentage with a view of how close trials came to failure, not just whether they crossed zero.

Scope for the plan to address:

    Define and propose a "near-miss" threshold (e.g. % of trials that dropped below some absolute dollar figure or some percentage of the starting bridge balance at any point) — this needs a sensible default but should be a configurable threshold, not hardcoded, consistent with this tool's existing "no silent defaults" standard
    If any trials do fail (currently 0% in recent runs, but this must work correctly for households where it isn't 0%), report the age distribution of failures and, where feasible, attribute a primary driver (poor early sequencing, a large CGT event, mortgage stress) — check what's actually derivable from existing tracked state versus what would need new instrumentation
    Confirm how this interacts with the existing "worst simulated outcome" single-trial display — is this an extension of that feature or a separate summary block

Dependency: Relies on the same per-trial, per-age tracking as Work Item 1 if the near-miss threshold is checked at every age rather than only at the final running minimum — confirm this in the plan and scope the two together if so.
Work Item 3 — Sequencing risk isolation

Requirement: Produce a direct measure of sequencing risk — the effect of poor returns landing early in the bridge versus late — rather than leaving it implicit in the aggregate Monte Carlo spread.

Scope for the plan to address:

    Propose a concrete method: e.g. re-running the simulation with the worst N years of a trial's return sequence deliberately front-loaded vs back-loaded, and comparing success probability / P5 outcome between the two. Confirm this is methodologically sound and describe exactly how "worst years" would be identified and reordered without breaking the correlation structure between equity/super/inflation returns established via the Cholesky generation approach used elsewhere in this tool
    This is likely the most build-intensive item in this phase — flag clearly if it requires running additional simulation passes beyond the standard trial count (i.e., does this need its own dedicated run, doubling compute time, or can it be derived from the existing trial set post-hoc)
    Propose how this is displayed — likely a simple before/after comparison (two success percentages, two P5 figures) rather than a full trajectory, to keep it legible

Dependency: None on other items, but confirm whether it can reuse the per-age data capture from Work Item 1 to identify which years were "worst" within each trial, or whether it needs independent instrumentation.
Work Item 4 — Drawdown source composition

Requirement: Report where the money funding the bridge actually came from — offset/cash, non-offset asset sales (with associated CGT), and any other source — not just the resulting balance.

Scope for the plan to address:

    Confirm what's already tracked per-trial in _drawdown() that could feed this (offset draws, asset sales via sell_assets()) versus what needs new accumulation — likely a running total per source per trial, aggregated to median/P5/P95 across trials at the end
    Display as a simple composition table: median $ drawn from offset/cash, median $ drawn from investment sales, over the whole bridge period
    Confirm whether this should also break out by offset reserve mode if a non-default mode (stall_prevention / interest_cancelling) was used, since that was specifically built to change this composition and currently has no visibility in the output at all

Dependency: Natural pairing with Work Item 5 (tax breakdown), since CGT paid is a direct byproduct of the investment-sale component of this same composition data. Scope and likely build together — same underlying data capture point in _drawdown().
Work Item 5 — Tax breakdown (CGT)

Requirement: Report total CGT paid over the bridge period, with the 30% floor's specific impact made visible, given this was a major recent build (Phase 1 + Phase 2 CGT reform) with currently zero visibility in the output.

Scope for the plan to address:

    Median (and P5/P95) total CGT paid across the bridge period, household total
    Per-earner CGT breakdown where ownership splits are non-default (multi-earner, jointly-held accounts) — connects to the ownership field work already shipped
    A specific comparison figure: CGT actually paid (with the 30% floor applied) vs what CGT would have been at pure marginal rates with no floor — this isolates and quantifies the floor's cost, which is a genuinely new piece of law and worth making visible to a client
    Confirm data capture: does sell_assets() currently return or accumulate CGT paid per call in a way that can be summed per trial, or does this need a new accumulator threaded through _drawdown()

Dependency: Build alongside Work Item 4 (shared data capture point). Also depends on Work Item 7 (real/nominal labelling) being resolved for how these dollar figures are presented — confirm whether CGT figures should be shown in nominal (as-paid) terms or real terms, and be explicit about which.
Work Item 6 — Scenario comparison table

Requirement: Show the client the effect of specific strategic choices already built into this tool but not currently surfaced as a comparison — at minimum: with vs without part-time income, and offset fully drained vs a protected reserve floor.

Scope for the plan to address:

    Confirm mechanism: does this run the full Monte Carlo simulation twice (or more) per comparison — once with the client's actual configuration, once with a counterfactual toggle changed — and report the swing in success probability and P5 horizon balance? Confirm compute cost implications (each comparison row is a full additional 10,000–50,000 trial run) and propose whether this should be opt-in (a separate menu option, not run by default on every simulation) given the cost
    Propose the initial comparison set: "current plan" vs "no PT income" vs "offset drained fully (no reserve floor)" vs "offset reserve floor at [client's configured value]" — confirm which of these are genuinely meaningful given the client's actual inputs (e.g. a client with no mortgage has no reserve-floor comparison to make)
    Display as a compact table: scenario label, success probability, P5 horizon balance, delta from the base case

Dependency: None technically, but this is likely the most expensive item to run (multiple full simulations) — sequence this last, after Work Items 1–5 establish what data is worth comparing across scenarios in the first place.
Work Item 7 — Real vs nominal labelling across all output

Requirement: Every dollar figure in the results output must be explicitly labelled as real (today's dollars) or nominal (then-year dollars), given inflation is a live simulation input and this is currently unstated anywhere in the output.

Scope for the plan to address:

    Audit every existing dollar figure in the current output (horizon percentiles, worst-case, super balances, mortgage remaining) and determine which basis each is currently calculated in — confirm this isn't already inconsistent internally (e.g. some figures accidentally nominal, others accidentally real, without anyone having specified which was intended)
    Propose a single consistent convention for this tool (recommend defaulting to real/today's-dollars throughout, since that's more interpretable for a client comparing to their current cost of living, but confirm with the firm) and a clear label on every table/section header (e.g. "(today's dollars)" or "(as at age 60, inflation-adjusted)")
    This is comparatively low-effort and should not block on other items — flag as safe to implement early or in parallel, since it's primarily a labelling and consistency pass rather than new data capture

Dependency: None — lowest-risk, lowest-effort item in this phase. Recommend the agent complete this first as a quick, low-risk win, and to establish the real/nominal convention before Work Items 1, 4, 5, and 6 add new dollar figures that would otherwise need to be retrofitted with the same labelling shortly after.
Work Item 8 — UX and navigation for the expanded results (CLI)

Requirement: Work Items 1–6 each add a new, potentially large block of output (trajectory tables, near-miss analysis, sequencing comparison, drawdown composition, tax breakdown, scenario comparison). None of this should be dumped onto the screen unconditionally after every run — the current single summary screen is already dense. Design a navigable results flow: the initial summary screen stays as the default landing view, with a way to drill into any of the new reports individually and return to the summary without re-running the simulation.

Scope for the plan to address:

    Landing view stays the current summary. After the 10,000/50,000-trial run completes, the existing summary screen (success probability, horizon percentiles, worst case, super, mortgage) remains the first thing shown — none of the new items should be auto-appended to it by default, or the screen becomes unreadable.
    A menu of additional views, presented after the summary. Propose a simple post-summary prompt, e.g.:

  View additional detail?
    1. Age-by-age trajectory
    2. Near-miss / failure depth analysis
    3. Sequencing risk comparison
    4. Drawdown source composition
    5. Tax (CGT) breakdown
    6. Scenario comparison
    7. Earliest feasible retirement age
    0. Back to summary
    q. Exit
  Choice:

Confirm this is buildable within the existing CLI framework (check what prompt/menu utilities are already used elsewhere in ui.py, e.g. _prompt_int, and reuse that pattern rather than introducing a new interaction style).

    Navigation back to the summary or menu, not just forward. Every detail view needs a way to return without exiting the results flow entirely — confirm whether a literal ESC keypress is feasible in this CLI's current input handling, or whether a typed option (e.g. "0" or "b" for back, consistent with existing prompt conventions in this tool) is the more realistic mechanism given the terminal/input library in use. State clearly in the plan which is actually achievable — don't commit to an ESC-key interaction if the underlying input handling doesn't support it, propose the closest equivalent instead.
    Re-entrant, not re-simulating. Moving between views must not re-run the Monte Carlo simulation — all results need to be held in memory (or read from the already-computed SimulationResults object) for the duration of the results-viewing session, so switching between trajectory / tax breakdown / scenario comparison is instant, not another multi-second computation. Flag this explicitly for Work Item 6 (scenario comparison) in particular, since that item already involves running additional simulations — those additional runs should happen once, on first selecting that menu option, and be cached for the rest of the session rather than recomputed every time the user navigates back to it.
    Exit behaviour. Confirm what happens on exit from the results flow — return to the main configuration menu (to run another simulation) vs exit the program entirely — and that this is consistent with how the tool currently behaves at the end of a run.
    Legibility per view. Each detail view should be able to stand alone on screen without requiring the user to have the summary visible simultaneously (i.e. don't assume scrollback — repeat any essential context, like the client's name/scenario label, at the top of each detail view if the CLI doesn't support a persistent header).

Dependency: This item wraps all of Work Items 1–6 — it should be scoped last, once the actual shape and size of each new report is known from the other items' plans, but the navigation pattern itself (menu, back option, caching) can be prototyped early against placeholder content if useful for early validation.
Work Item 9 — Earliest feasible retirement age

Requirement: Given the client's plan otherwise unchanged (spending, mortgage, contributions, all other inputs held fixed), find the earliest retirement age that still clears the success threshold — e.g. client enters retirement age 50, tool reports "this plan also succeeds at age 45; age 44 fails." This is a search over retirement age only — not a general optimiser, not a spending solver. Everything else in the client's configuration stays exactly as entered.

Scope for the plan to address:

    Search method. Propose a concrete search strategy — a simple linear scan (try 50, 49, 48… down to some floor, stop at the first age below threshold) is easiest to reason about and explain to a client, but expensive if the range is wide (each age tried is a full trial run). A binary search is faster but requires the relationship between retirement age and success probability to be monotonic (earlier retirement should always be harder or equal, never easier, for a fixed spending/mortgage/contribution plan) — confirm this monotonicity assumption actually holds given how this tool's mechanics work (e.g. does a longer accumulation phase always help, or could an edge case like PT income timing or mortgage term interactions break monotonicity?). State clearly which method is proposed and why, and flag if a linear scan is safer given the tool's own complexity.
    Floor for the search. The search needs a lower bound so it doesn't search indefinitely — propose a sensible floor (e.g. don't search below age 40, or don't search below whatever age would leave less than some minimum number of working years) and confirm this should be configurable or firm-set, not hardcoded silently.
    Threshold consistency. This must use the same configurable success threshold from the earlier "configurable success threshold" work (Option A build, already shipped) — the earliest feasible age is only meaningful relative to a defined pass bar, and that bar should be whatever the user has it set to, not a separately hardcoded 95%.
    What "otherwise unchanged" means precisely for multi-earner households. If there are two earners, confirm whether this searches one earner's retirement age with the other held fixed, both simultaneously (and if so, how — same age, or independently optimised), or whether this feature is scoped to single-earner households only for an initial build. This needs an explicit decision, not an assumption — flag it clearly as a question for the firm.
    Reporting the result. Show the earliest passing age and the success probability at that age, plus (for context) the success probability at the client's originally-entered age, so the client can see both figures side by side — e.g. "At your stated retirement age (50): 100% success. Earliest feasible age given this plan: 45 (96% success)." If a linear scan is used, also show the first failing age immediately below the reported feasible age, so the client can see how close to the edge the answer sits, not just the answer itself.
    Compute cost and caching. Each age tried is a full trial run — state the expected number of runs for a reasonable search range (e.g. a 10-year linear scan from 50 down to 40 is 10 full runs) and confirm this is only ever triggered on demand (see Work Item 8's opt-in menu), never automatically, and that the result is cached for the results-viewing session once computed rather than re-run if the user navigates back to it.
    Caveat that must ship with the result. This answer changes if the client changes any other input (spending, mortgage payoff, PT income assumptions) — the output must state plainly that this is the earliest age for this specific plan configuration, not a general result, so it isn't misread as a standalone guarantee independent of everything else entered.

Dependency: Add as an option on the Work Item 8 results menu (opt-in, triggered on demand — see updated menu list below). Reuses the configurable success threshold already shipped. No dependency on Work Items 1–7's content, though it benefits from Work Item 7's real/nominal labelling convention for any dollar figures shown alongside the result.
Work Item 10 — Mortgage amortisation table with offset-neutral tracking

Requirement: A year-by-year amortisation table for each mortgage, from the simulation's starting year through to the user-specified loan_term_end_age (already an existing field per the earlier mortgage term-clearance work). Where offset accounts are linked to the mortgage, the table must clearly mark the year the offset balance reaches or exceeds the outstanding mortgage balance ("mortgage neutral" — i.e. the point from which the offset overflow sweep, already built, begins diverting surplus into investment accounts), and show that sweep amount for every year from that point onward.

Scope for the plan to address:

    Table structure. Propose columns per year: age / calendar year, opening mortgage balance, interest charged, principal paid, closing mortgage balance, offset balance (if any linked offset exists), and — from the mortgage-neutral year onward — the amount swept from offset into investment accounts that year. Confirm this can be built as a per-trial or a percentile-aggregated view (median/P5/P95 mortgage balance by year across all trials) — recommend the plan address both: a median-path table as the primary view (single clean table, most representative), with percentile bands available as a secondary/optional view given mortgage amortisation is comparatively less variable trial-to-trial than investment balances (interest rate is currently static per the earlier finding, so the main source of variance here is how much offset gets diverted to living expenses under a poor investment sequence, not the mortgage mechanics themselves — confirm this reasoning holds before committing to median-only as sufficient).
    Data capture. Confirm what's already tracked per year in amortize_mortgage_monthly() / _drawdown() that can feed this directly (mortgage balance, offset balance are almost certainly already state variables) versus what needs new accumulation — likely just retaining the existing per-year state into a returned trajectory rather than discarding it after each year, similar in shape to Work Item 1's trajectory capture. Flag if this can share infrastructure with Work Item 1 (age-by-age trajectory) given both are fundamentally "retain per-year state across the simulation horizon" problems — scope together if so.
    "Mortgage neutral" definition and detection. Confirm the exact comparison: offset balance ≥ outstanding mortgage principal (not the original loan amount) at the end of a given year, or checked continuously/monthly within the year given the tool's monthly amortisation stepping. State clearly which resolution is used and why, consistent with the granularity discussion already had regarding annual vs monthly tracking elsewhere in this tool. Mark this year distinctly in the table output (e.g. a highlighted row or an inline label "← mortgage neutral").
    Sweep display from the neutral year onward. The offset overflow sweep mechanism already exists (hardcoded, not configurable, per the earlier finding). Confirm the table surfaces the actual swept amount per year from handle_offset_overflow() (or wherever that logic lives) rather than recalculating it separately — this must read the same value the simulation actually used, not an approximation, to avoid the same "two sources of truth diverging" problem that caused the earlier success-probability/worst-case contradiction bug.
    Multiple mortgages. If a household has more than one mortgage, confirm whether this is one table per mortgage, or a combined household view — recommend one table per mortgage (clearer), with the offset-neutral and sweep columns only appearing for a mortgage that actually has a linked offset account (per the existing mortgage.offset_accounts structure).
    Interaction with offset reserve mode. If a non-default offset reserve mode is set (stall_prevention or interest_cancelling, from the earlier work), confirm the table reflects the actual constrained drawdown behaviour under that mode — e.g. under interest_cancelling, offset may never be available to sweep into investments at all before the mortgage is fully repaid, since the floor locks the full principal. State this interaction explicitly in the plan rather than assuming the sweep behaves identically regardless of reserve mode.
    Real vs nominal. Confirm this table follows the same labelling convention established in Work Item 7 — mortgage balances and interest figures should be clearly marked as nominal (actual dollar amounts owed/paid in that year) rather than inflation-adjusted, since a mortgage balance is inherently a nominal figure (the debt itself doesn't inflate), and mixing conventions here would be actively misleading rather than just inconsistent.
    Display. As with other detail items, this is a candidate for the Work Item 8 results navigation menu (e.g. "Mortgage amortisation") rather than appended to the main summary — confirm and add to that menu's option list.

Dependency: Benefits from sharing trajectory-capture infrastructure with Work Item 1 if built concurrently — flag this explicitly in the plan rather than building two separate per-year state retention mechanisms. Depends on Work Item 8 for its place in the navigation menu, and Work Item 7 for labelling convention.

    Work Item 7 first — cheapest, establishes a convention the other items should follow rather than retrofit.
    Work Items 1 + 2 + 10 together where practical — all three fundamentally need per-year trajectory data retained across the simulation horizon (bridge assets, near-miss thresholds, mortgage/offset balances); scope the underlying data capture once and feed all three views from it rather than building three separate retention mechanisms.
    Work Items 4 + 5 together — shared data capture (drawdown/CGT accumulation in _drawdown()).
    Work Item 3 — isolated, more complex, benefits from Item 1's data model being settled first.
    Work Item 6 — expensive (multiple simulation runs), most benefits from knowing what Items 1–5 already surface before deciding what's worth comparing.
    Work Item 9 — also expensive (multiple simulation runs via search), can be scoped alongside Work Item 6 since both involve repeated full simulations and similar caching considerations, but is functionally independent of it.
    Work Item 8 last overall — the navigation/menu shell around everything above, including Work Item 9 and Work Item 10 as menu options; scope once the actual shape of each report is known, though the menu pattern itself can be prototyped early if useful.

What the agent should deliver for this planning task

For each of the 10 items: a short implementation plan (data model changes, display changes, test approach), an effort estimate (relative — small/medium/large), explicit dependencies on other items in this list, and any question that needs a decision from the firm before implementation can start (e.g. near-miss threshold default, real-vs-nominal convention, whether scenario comparison should be opt-in given compute cost, whether ESC-key navigation is technically feasible in the current CLI input handling, whether the earliest-retirement-age search supports multi-earner households in its first build, whether the monotonicity assumption needed for a binary search actually holds for this tool's mechanics, and whether the mortgage amortisation table should default to a median-path view or offer percentile bands from the outset).

Do not write implementation code as part of this task. This is a planning and scoping deliverable only.

