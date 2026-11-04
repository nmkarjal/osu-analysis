# how it works

basically my notes on what the code does, including the stuff that didnt work. might be a bit messy

## data

`pull` grabs your profile, top plays (up to 200), recent attempts including fails, most played maps, and difficulty for each map with mods applied. the api only gives 24 hours and 100 attempts of recent stuff, so every pull gets added to `history/recent_all.csv`

random things i ran into:

- failed scores come back with id 0, so i key those on map + timestamp. before that all the fails collapsed into one row lol
- the api gives nomod ar and od, so those get worked out from the mods

two tables get used everywhere:

- **passes** - top plays + logged passes. top plays are picked by pp though, which flattens everything (a hard map only makes the list if the play went well)
- **attempts** - every logged attempt with 50+ objects hit, fails included. less biased, but a fail only covers the part you played

## ceilings

- aim ceiling = the aim value where you average 3 misses per 100 objects, with speed at your median
- speed ceiling = the speed value where you average 15 100s/50s per 100 objects, with aim at your median

those two together are roughly 86% acc. it fits a plane on your last 250 attempts and gets the ranges by resampling whole maps. the "limiter" is whichever line you hit first going up along your usual speed/aim ratio

first i tried a "weakest link" model that got one ceiling per skill straight from pass/fail. sounded cool, gave dumb numbers. aim and speed are too correlated in real plays and it wasnt any better than plain logistic regression so i removed it

## finish chance

three models compete: your base rate, logistic on stars, logistic on aim + speed + log length (ridge, nf plays dropped)

test is walk-forward: train on the past, remove the test maps from training, predict the next chunk. a model only counts as trusted if its gain over the base rate has a 95% interval above zero (resampling maps). otherwise the report says finish chance is a rough guess

if your client submits restarts as fails then "finish" is partly just how often you restart. thats why map roles use predicted acc and not this

## acc, misses, 100s

ridge regressions on aim, speed, effective od and log length, checked with 5 fold cv grouped by map against guessing your average. if one doesnt beat that by 3% its not used and map picking falls back to a plain star rating fit

## weakness scan

each check regresses one error thing on one factor, with aim, speed and their squares as controls. stuff that actually mattered:

- **clustered errors by map** - retries of the same map arent independent and normal p values come out way too small
- **squared difficulty terms** - errors curve up with difficulty. without these anything that follows difficulty picks up the curve and looks like its own effect. gave me a fake "low ar hurts you" result
- **collinearity check** - if aim and speed explain over 75% of a factor it says it cant judge it instead of testing it
- **benjamini-hochberg at 10%** over all the checks
- **session effects need 5+ sessions**

on simulated noise 12% of scans flagged something, target was about 10%, so close enough

weaknesses get ranked by the end of their interval closest to zero, in acc points (a miss is 1pp per 100 objects, a 100 about 0.7). that way a big unsure effect doesnt beat a smaller sure one

habits are separate since theyre just what you do, not a tested effect: finishing under half your attempts, tons of early quits, long fail streaks on one map, and whether instant retries actually get you further

## sessions

30 min gap = new session. each session gets adjusted to equal difficulty so playing harder maps doesnt look like getting worse. trends only show up with 5+ sessions over 3+ days

## picking maps

1. star range comes from your acc fit (where it crosses 95% and 76%)
2. search your sources from the config, filter per difficulty. kept for 14 days
3. shortlist the most favourited sets per 0.5 star band so every level has options, get attributes for those (cached 30 days)
4. predict acc + finish chance and score each map per role:

| role | target acc | also wants |
|---|---|---|
| warm-up | 95% | |
| aim focus | 88.5% | lower speed/aim than your usual maps |
| speed focus | 88.5% | higher speed/aim than your usual maps |
| stretch | 80% | |
| dt | 88.5% | dt attributes, bpm cap, no `no_dt` sources |

5. pick greedily with a penalty for repeat artists and near identical difficulty, no mapset twice

top play maps get skipped. maps youve already tried go in a revisit queue: unfinished ones you got 40%+ through come back after a day, finished ones under 90% come back after 2, 4, 8, then 16 days

## plans

warm-up, one or two focus blocks (top skill areas from the scan, or your limiter if nothing showed up), revisit or stretch, cool-down. goals are 10% better than your recent sessions at equal difficulty

plans get saved and checked next run. if finished plays on planned maps land 2+ points over the prediction (6+ plays), the next plan's target acc drops by 1, and the other way around, capped at 6 either way. each plan only counts once

## not done

- replays. hit errors and per pattern stuff would be way more useful than all of this but the api doesnt give it
- proper mixed models. clustered ols is simpler and fine for now
- checking if the plans actually make anyone better. would need way more players than just me
