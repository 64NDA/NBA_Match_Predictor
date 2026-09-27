# How the predictor works, in plain English

The technical version is in [MODEL_V3_DESIGN.md](MODEL_V3_DESIGN.md). This one explains the same ideas with
no statistics background needed. Numbers in the examples are made up to illustrate the idea.

---

## The big picture: how would a smart fan guess tonight's winner?

A smart fan asks four questions:

1. **Who's been winning lately, and by how much?** → *Elo*
2. **Who's actually playing tonight, and how good are they?** → *Player ratings + injuries*
3. **Where is the game, and is anyone tired?** → *Home court, rest, travel*
4. **Do these two teams' styles clash?** → *Matchup* (we tested this carefully; see Step 3 for what we found)

Our predictor answers all four with numbers, **in points**, then adds them up.

> **Example: Celtics at Lakers**
> | Question | Answer | Points |
> |---|---|---|
> | Home court | Lakers at home | Lakers **+1.8** |
> | Recent results (Elo) | Celtics have been better | Celtics **+2.5** |
> | Who's playing | Lakers' star is out tonight | Celtics **+4.0** |
> | Rest | Celtics played last night | Lakers **+1.0** |
> | **Total** | | **Celtics by 3.7** |
>
> Then "Celtics by 3.7" is turned into "Celtics win 62% of the time", plus a predicted score like 113–109.

Every bar in the app's **"Why?"** chart is one row of that table.

---

## Step 0: collecting the data (what we download)

All of it comes free from the NBA's own stats website:

| What | Think of it as | Example |
|---|---|---|
| **Game results** | the scoreboard | "Celtics 112, Lakers 109" |
| **Box scores** | each player's stat line | "Tatum: 30 pts, 8 reb, 36 minutes" |
| **Rotations** | who was on the court, and when | "Tatum played 0:00–9:12 and 14:30–24:00" |
| **Play-by-play** | a diary of every play | "7:41, Q1: Tatum makes a 3. Score 12–9" |
| **Injury reports** | the NBA's official "who might sit" list | "LeBron James: Questionable (ankle)" |
| **Rosters** | who is on each team today | the 15-man squad lists |

That covers 11 seasons, about 14,500 games. It's saved on your computer so we only download it once.

---

## Step 1: Elo, the "recent results" score

Think of **Elo like a video game ranking**. Every team starts at 1505 points.
- Win a game and you **take points** from the loser.
- Beat a strong team and you take **more**. Beat a weak team and you take **fewer**.
- Win by 30 and you take more than for winning by 2.

After thousands of games, good teams sit high (e.g. 1700) and bad teams low (e.g. 1300).

**Over the summer,** every team drifts a bit back toward average, because rosters change. **New in v3:** the
drift heads toward how good the team's *new roster* looks, not just "average". A team that signed a superstar
starts the season higher.

**Elo's weakness:** it only knows *results*. It doesn't know your best player broke his ankle yesterday. That's
why we need Step 2.

---

## Step 2: players, the heart of the model

### 2a. How good is each player? ("impact rating")

Points scored don't tell the whole story. A great defender might score 6 points but stop the other team
scoring 10. So we measure impact like this:

> **Watch what happens to the score while each player is on the court.**

We cut every game into **stints**: stretches of time where the same 10 players are on the floor. For example:

| Stint | Celtics on court | Lakers on court | Score during stint |
|---|---|---|---|
| 1 | Tatum, Brown, White, Holiday, Porziņģis | LeBron, Davis, Reaves, Russell, Hachimura | Celtics +4 |
| 2 | Tatum, Brown, *bench guy*, Holiday, Horford | LeBron, *bench guy*, Reaves, Russell, Hachimura | Lakers +2 |
| … thousands more … | | | |

Across **hundreds of thousands of stints**, a computer works out which players are always on court when their
team does well, *after accounting for who they played with and against*. That's a **ridge regression**, a
standard machine-learning method. Its output for each player is two numbers:

- **Offence rating:** "when he's on court, his team scores X more points per 100 plays than average"
- **Defence rating:** "when he's on court, the other team scores X fewer"

This is how **defence gets counted**, which box scores are bad at.

**Problem:** a player with only 50 minutes of court time has too few stints to judge. So we also give every
player a **starting guess from his stat line** (points, assists, rebounds, steals and so on), and let the
stint data move him away from that guess as he plays more. Rookies start at "average bench player".

We **re-calculate every week**, so ratings follow form.

### 2b. How many minutes will each player play?

A star who plays 36 minutes matters far more than a bench player who plays 8.
- Each player's **usual minutes** are his recent average.
- When someone is **out**, his minutes are **handed to teammates**, mostly to the players who usually play a
  lot. The team always adds up to 240 minutes (5 players × 48 minutes).
- **In the playoffs, coaches shorten the bench**, so stars play more. The model learns how much from past playoffs.

### 2c. Will each player actually play?

Each player gets a **chance of playing**, from 0% to 100%, from the best information available:

1. **You**, in the app's Matchup explorer: "I know he's out."
2. **The official injury report.** We learned from history what each label *really* means. For example, if
   "Questionable" players actually play about half the time, Questionable = 50%.
3. **Recent history, if he isn't on the report.** For example, "he's missed the last 12 games" = very unlikely to play.

### 2d. Putting it together: how strong is the team *tonight*?

> **Team strength = add up (each player's rating × his expected minutes)**

This is why **star power happens automatically**: a star has a big rating *and* big minutes, so he counts
the most. When he's out, his minutes go to a worse player, and the team's number drops by exactly the
difference. The app can say things like **"Lakers are 4.8 points worse without Dončić."**

We also estimate **pace** (how fast both teams play) to predict how many points each side will score.

---

## Step 3: matchups, "does their style beat ours?"

Some teams are bad matchups for each other even when they're equally good overall.

| Style clash | Example |
|---|---|
| **Rebounding battle** | A team that grabs lots of its own misses, against a team that's bad at boxing out |
| **Three-point shooting** | A team that lives on threes, against a defence that's great at stopping them |
| **Attacking the rim** | A team that drives to the basket, against a team with a giant shot-blocker |
| **Size** | A tall lineup against a small one |
| **Playoff series** | By game 4 of a series, coaches have adjusted. How has each team done so far *compared with what we expected*? |

We describe each team's style with numbers (how often they shoot threes, how often they get to the rim, how
well they rebound…). Then we let **machine learning (a random forest / gradient boosting)** look for patterns
like "rim-attacking teams under-perform against shot-blockers". This is the kind of job those models are
good at: **spotting combinations** we might not think to write down.

**What testing showed:** over the regular season these style features made predictions slightly *worse*, so
they're **not in the final model**. In the playoffs they showed a small hint of helping, but 361 playoff games is
too few to be sure. The code is kept, to test again as more playoffs are played. Once you know how good each
player is and who's playing, most "style" is already accounted for.

---

## Step 4: where, and how tired

- **Home court:** about +1.8 points for the home team. It's been shrinking since around 2019, and the model
  learns the current size.
- **Neutral venues** (e.g. Paris, Mexico City, the NBA Cup in Las Vegas): no home bonus.
- **Some arenas are extra hard**, like Denver's mile-high altitude. Tested separately.
- **Playoffs:** home court counts a bit more.
- **Tiredness:** second night of a back-to-back, three games in four nights, long flights, crossing time zones.

---

## Step 5: the final prediction

A **combiner** (another ridge regression) looks at 11 seasons of games and learns **how much to trust each
piece**. For example: "Elo is worth this much, player ratings this much, the matchup this little". It gives:

1. **Predicted margin:** "Celtics by 3.1"
2. **Predicted total points:** e.g. 221
3. **Predicted score:** split the total using the margin → **Celtics 112, Lakers 109**
4. **Win chance:** a margin of 3.1 points becomes about 59%. Close games are coin-flip-ish, blowouts aren't,
   and we measured exactly how close from history.

Because the score and the win chance come from the **same margin**, they can never disagree. That fixes the
v1 problem where the model sometimes said *both* teams would win.

### Where machine learning is used
| Job | Method | Plain English |
|---|---|---|
| Player impact | Ridge regression | "Who's on court when good things happen?" |
| Chance a player plays | Logistic regression vs gradient boosting (we keep the better one) | "What usually happens to players in this situation?" |
| Minutes redistribution | Simple rule vs gradient boosting | "Who usually soaks up the missing minutes?" |
| Style matchups | Random forest / gradient boosting | Tested: didn't help, so not used |
| Second opinion | Gradient boosting, a small neural network | Tested: no better than ridge, so not used |
| Final combiner | Ridge regression | "How much should we trust each clue?" |

The lesson: fancy models help most where there's lots of data and the pattern is bendy. "Will he play?" has
475,000 examples, and missing 1 game is very different from missing 10. Game results have about 13,000
examples and mostly add up in a straight line.

---

## How we know it works (and aren't fooling ourselves)

**Testing.** We pretend it's the past. To test the 2023-24 season, the model only sees games from *before*
2023-24, then predicts 2023-24 blind. We repeat this for 4 seasons. It's like marking a student on an exam
they haven't seen.

**A locked exam.** The 2025-26 season is kept locked away and used **once**, at the very end, to check we
didn't just get lucky tuning on the other seasons.

**No peeking.** Automatic tests check that no prediction ever uses information from after the game started,
such as an injury report published after tip-off.

**Three honesty levels.** We report accuracy:
- **(a)** if we magically knew who'd play: the best case
- **(b)** using the injury report as it stood 1 hour before tip-off: **the real, honest number**
- **(c)** with no injury report at all

**Scorecard (live: using the injury report an hour before tip-off):**
| Version | 2021-22 → 2024-25 | 2025-26 (locked exam) |
|---|---|---|
| v1 (random forest, one season) | ~62%, and self-contradicting | – |
| v2 | 65.1% | 68.4% |
| **v3** | **67.1%** | 68.8% |
| Elo alone | 64.8% | 68.2% |
| Bookmakers | ~68–70% | |

v3's **percentages** are clearly better than v2's in both periods. On 2025-26, a season where even simple Elo got
68%, its **pick accuracy** was about the same as v2's (68.8% vs 68.4%, within luck).

**Who's on which team, and when.** Every prediction uses the roster as it stood on that day. We replay the
NBA's list of every trade, signing and release since 2015, so a player traded in February stops counting for his
old team that day. For today's games we use the official rosters.

Basketball has a lot of luck in it. Even the bookmakers, with teams of analysts, get about 3 in 10 wrong.

---

## What happens to the data, start to finish

```
NBA website ──download──▶ raw files (saved, never re-downloaded)
                               │
                               ▼
            tidy tables: games · player stat lines · stints · injury reports
                               │
       ┌───────────────┬───────┴────────┬──────────────────┐
       ▼               ▼                ▼                  ▼
     Elo         player ratings    who plays &        team styles
  (results)     (weekly refit)     for how long     (shooting, size…)
       │               │                │                  │
       └───────────────┴──── features ──┴──────────────────┘
                   (all in "points", home minus away)
                               │
                               ▼
                 combiner → margin + total → win % + score
                               │
                               ▼
                  app: predictions, "Why?", what-if injuries
```

## Glossary
- **Margin:** how many points the home team wins by (negative = loses).
- **Per 100 possessions:** a way to compare players and teams fairly, whether their games are fast or slow.
  A possession is one team's turn with the ball.
- **Regression:** a method that finds the best numbers to explain past results.
- **Ridge regression:** a regression that's deliberately cautious, which stops it over-reacting to small samples.
- **Random forest / gradient boosting:** machine-learning methods that build hundreds of small "if this, then
  that" decision trees. They're good at finding combinations.
- **Log-loss:** a score for how good the *percentages* are, not just the picks. Lower is better. It punishes
  confidently wrong predictions.
- **Holdout:** data kept locked away for one final, honest test.
